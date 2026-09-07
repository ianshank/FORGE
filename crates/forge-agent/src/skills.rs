//! Hierarchical skill executor over reusable primitive-action families.
//!
//! High-level policies select a [`SkillsConfig`] option (Sutton options /
//! HRL primitives). This module executes the option as a low-level
//! [`crate::baselines::Agent`], terminating on horizon expiry or an
//! initiation-set miss (e.g. aerial skills on ground agents).

use forge_core::WorldState;
use forge_types::skill::{SkillCategory, SkillsConfig};
use forge_types::Action;
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use tracing::{debug, info, instrument, warn};

use crate::baselines::{Agent, GreedyNavigator, HeuristicAgent, RandomAgent};

/// Hierarchical agent that selects catalog skills then emits primitives.
pub struct HierarchicalSkillAgent<R: Rng + Send> {
    catalog: SkillsConfig,
    comm_vocab_size: u16,
    active_index: Option<usize>,
    steps_in_skill: u32,
    navigator: GreedyNavigator,
    gatherer: HeuristicAgent<R>,
    explorer: RandomAgent<R>,
}

impl HierarchicalSkillAgent<Pcg64Mcg> {
    /// Builds a deterministic hierarchical agent from a catalog and seed.
    #[instrument(skip(catalog))]
    pub fn seeded(catalog: SkillsConfig, comm_vocab_size: u16, seed: u64) -> Self {
        let mut split = Pcg64Mcg::seed_from_u64(seed);
        let gather_seed: u64 = split.gen();
        let explore_seed: u64 = split.gen();
        Self::new(
            catalog,
            comm_vocab_size,
            Pcg64Mcg::seed_from_u64(gather_seed),
            Pcg64Mcg::seed_from_u64(explore_seed),
        )
    }
}

impl<R: Rng + Send> HierarchicalSkillAgent<R> {
    /// Constructs a hierarchical agent from explicit RNGs.
    pub fn new(catalog: SkillsConfig, comm_vocab_size: u16, gather_rng: R, explore_rng: R) -> Self {
        info!(
            skills = catalog.skills.len(),
            default = %catalog.default_skill,
            enabled = catalog.enabled,
            "hierarchical skill agent initialized"
        );
        Self {
            catalog,
            comm_vocab_size,
            active_index: None,
            steps_in_skill: 0,
            navigator: GreedyNavigator::new(0, 0),
            gatherer: HeuristicAgent::new(gather_rng, comm_vocab_size),
            explorer: RandomAgent::new(explore_rng, comm_vocab_size),
        }
    }

    /// Currently executing skill id, if any.
    pub fn active_skill_id(&self) -> Option<&str> {
        self.active_index
            .and_then(|idx| self.catalog.skills.get(idx))
            .map(|spec| spec.id.as_str())
    }

    fn select_next_skill(&mut self, state: &WorldState, agent_idx: usize) -> Option<usize> {
        let can_fly = state
            .agents
            .get(agent_idx)
            .map(|agent| agent.capabilities.can_fly)
            .unwrap_or(false);
        let n = self.catalog.skills.len();
        if n == 0 {
            warn!("skill catalog is empty; emitting Noop");
            return None;
        }
        let start = self
            .active_index
            .map(|idx| (idx + 1) % n)
            .or_else(|| {
                self.catalog
                    .resolve_default()
                    .and_then(|spec| self.catalog.skills.iter().position(|s| s.id == spec.id))
            })
            .unwrap_or(0);
        for offset in 0..n {
            let idx = (start + offset) % n;
            let spec = &self.catalog.skills[idx];
            if !spec.enabled {
                continue;
            }
            if spec.requires_drone && (!can_fly || !state.config.drone.enabled) {
                debug!(skill = %spec.id, "skipping drone-gated skill for ground agent");
                continue;
            }
            if spec.category == SkillCategory::Agriculture && !state.config.agri.enabled {
                debug!(skill = %spec.id, "skipping agriculture skill; agri disabled");
                continue;
            }
            if spec.category == SkillCategory::Communicate && self.comm_vocab_size == 0 {
                continue;
            }
            return Some(idx);
        }
        None
    }

    fn bind_navigator(&mut self, target_x: Option<u16>, target_y: Option<u16>, state: &WorldState) {
        let width = state.config.world.width;
        let height = state.config.world.height;
        let tx = target_x.unwrap_or(width / 2);
        let ty = target_y.unwrap_or(height / 2);
        self.navigator = GreedyNavigator::new(tx, ty);
    }
}

impl<R: Rng + Send> Agent for HierarchicalSkillAgent<R> {
    fn select_action(&mut self, state: &WorldState, agent_idx: usize) -> Action {
        if agent_idx >= state.agents.len() {
            debug!(
                agent_idx,
                "hierarchical skill agent: agent index out of range"
            );
            return Action::Noop;
        }
        if !state.agents[agent_idx].alive {
            return Action::Noop;
        }

        let needs_new = match self.active_index {
            None => true,
            Some(idx) => {
                let spec = &self.catalog.skills[idx];
                let horizon = self.catalog.effective_horizon(spec);
                self.steps_in_skill >= horizon
            }
        };
        if needs_new {
            match self.select_next_skill(state, agent_idx) {
                Some(idx) => {
                    let (id, category, horizon, target_x, target_y) = {
                        let spec = &self.catalog.skills[idx];
                        (
                            spec.id.clone(),
                            spec.category,
                            self.catalog.effective_horizon(spec),
                            spec.target_x,
                            spec.target_y,
                        )
                    };
                    info!(
                        skill = %id,
                        category = category.as_str(),
                        horizon,
                        "switching hierarchical skill"
                    );
                    self.bind_navigator(target_x, target_y, state);
                    self.active_index = Some(idx);
                    self.steps_in_skill = 0;
                }
                None => {
                    self.active_index = None;
                    return Action::Noop;
                }
            }
        }

        let idx = match self.active_index {
            Some(idx) => idx,
            None => return Action::Noop,
        };
        let remaining = {
            let spec = &self.catalog.skills[idx];
            self.catalog
                .effective_horizon(spec)
                .saturating_sub(self.steps_in_skill)
        };
        let action = {
            let spec = &self.catalog.skills[idx];
            let category = spec.category;
            let recipe_index = spec.recipe_index;
            let comm_token = spec.comm_token;
            match category {
                SkillCategory::Idle => Action::Noop,
                SkillCategory::Navigate => self.navigator.select_action(state, agent_idx),
                SkillCategory::Gather => self.gatherer.select_action(state, agent_idx),
                SkillCategory::Explore => self.explorer.select_action(state, agent_idx),
                SkillCategory::Craft => Action::Craft(recipe_index),
                SkillCategory::Combat => Action::Interact,
                SkillCategory::Communicate => Action::Communicate(comm_token),
                SkillCategory::Aerial => {
                    let altitude = state
                        .agents
                        .get(agent_idx)
                        .map(|agent| agent.altitude)
                        .unwrap_or(0);
                    if altitude == 0 {
                        Action::TakeOff
                    } else if remaining <= 1 {
                        Action::Land
                    } else {
                        Action::Hover
                    }
                }
                SkillCategory::Agriculture => Action::ScanMultispectral,
            }
        };
        self.steps_in_skill = self.steps_in_skill.saturating_add(1);
        debug!(
            skill = %self.catalog.skills[idx].id,
            step = self.steps_in_skill,
            ?action,
            "skill primitive selected"
        );
        action
    }

    fn name(&self) -> &str {
        "HierarchicalSkillAgent"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::config::ForgeConfig;
    use forge_types::constants;
    use forge_types::grid::Position;

    fn test_world() -> WorldState {
        let mut config = ForgeConfig::default();
        config.world.width = 16;
        config.world.height = 16;
        config.world.seed = constants::DEFAULT_SEED;
        config.agents.num_agents = 1;
        config.agents.comm_vocab_size = 0;
        config.task.max_episode_length = 64;
        WorldState::new(config).unwrap()
    }

    #[test]
    fn seeded_agent_is_deterministic() {
        let catalog = SkillsConfig::default();
        let seed = constants::DEFAULT_SEED.saturating_add(7);
        let mut a = HierarchicalSkillAgent::seeded(catalog.clone(), 0, seed);
        let mut b = HierarchicalSkillAgent::seeded(catalog, 0, seed);
        let mut world_a = test_world();
        let mut world_b = test_world();
        let ticks = constants::DEFAULT_SKILL_EXPLORE_HORIZON.saturating_mul(2);
        for _ in 0..ticks {
            let act_a = a.select_action(&world_a, 0);
            let act_b = b.select_action(&world_b, 0);
            assert_eq!(act_a, act_b);
            let _ = world_a.step(std::slice::from_ref(&act_a));
            let _ = world_b.step(std::slice::from_ref(&act_b));
        }
    }

    #[test]
    fn out_of_range_agent_index_is_noop() {
        let mut agent = HierarchicalSkillAgent::seeded(SkillsConfig::default(), 0, 1);
        let world = test_world();
        assert_eq!(agent.select_action(&world, 99), Action::Noop);
    }

    #[test]
    fn empty_catalog_emits_noop() {
        let catalog = SkillsConfig {
            skills: Vec::new(),
            ..SkillsConfig::default()
        };
        let mut agent = HierarchicalSkillAgent::seeded(catalog, 0, 1);
        let world = test_world();
        assert_eq!(agent.select_action(&world, 0), Action::Noop);
        assert!(agent.active_skill_id().is_none());
    }

    #[test]
    fn drone_skills_skipped_for_ground_agents() {
        let mut catalog = SkillsConfig::default();
        for spec in &mut catalog.skills {
            spec.enabled = spec.id == "aerial";
        }
        catalog.default_skill = "aerial".to_string();
        let mut agent = HierarchicalSkillAgent::seeded(catalog, 0, 1);
        let world = test_world();
        assert!(!world.agents[0].capabilities.can_fly);
        assert_eq!(agent.select_action(&world, 0), Action::Noop);
    }

    #[test]
    fn agriculture_skill_skipped_when_agri_disabled() {
        let mut catalog = SkillsConfig::default();
        for spec in &mut catalog.skills {
            spec.enabled = spec.id == "agriculture";
            spec.requires_drone = false;
        }
        catalog.default_skill = "agriculture".to_string();
        let mut agent = HierarchicalSkillAgent::seeded(catalog, 0, 1);
        let world = test_world();
        assert!(!world.config.agri.enabled);
        assert_eq!(agent.select_action(&world, 0), Action::Noop);
    }

    #[test]
    fn navigate_skill_emits_move_or_noop_toward_target() {
        let mut catalog = SkillsConfig::default();
        for spec in &mut catalog.skills {
            spec.enabled = spec.id == "navigate";
            if spec.id == "navigate" {
                spec.target_x = Some(15);
                spec.target_y = Some(15);
            }
        }
        catalog.default_skill = "navigate".to_string();
        let mut agent = HierarchicalSkillAgent::seeded(catalog, 0, 1);
        let mut world = test_world();
        world.agents[0].position = Position::new(0, 0);
        let action = agent.select_action(&world, 0);
        assert!(
            matches!(action, Action::Move(_) | Action::Noop),
            "navigate must stay in the move family, got {action:?}"
        );
        assert_eq!(agent.active_skill_id(), Some("navigate"));
    }

    #[test]
    fn dead_agent_is_noop() {
        let mut agent = HierarchicalSkillAgent::seeded(SkillsConfig::default(), 0, 1);
        let mut world = test_world();
        world.agents[0].alive = false;
        assert_eq!(agent.select_action(&world, 0), Action::Noop);
    }
}
