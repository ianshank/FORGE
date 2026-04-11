//! Scenario registry: index, search, and retrieve scenarios.
//!
//! The [`ScenarioRegistry`] loads scenario configs from a directory and
//! provides search by tag, tier, agent count, and text query.

use std::collections::HashMap;
use std::path::Path;

use tracing::{debug, instrument, warn};

use crate::config::ScenarioConfig;

/// Search query for filtering scenarios.
#[derive(Debug, Clone, Default)]
pub struct ScenarioQuery {
    /// Filter by tags (any match).
    pub tags: Vec<String>,
    /// Filter by tier range (inclusive).
    pub tier_range: Option<(u8, u8)>,
    /// Filter by minimum agent count.
    pub min_agents: Option<u32>,
    /// Text search in name/description.
    pub text_search: Option<String>,
}

/// Index entry for quick lookups.
#[derive(Debug, Clone)]
struct IndexEntry {
    /// Scenario ID.
    id: String,
    /// Tags for search.
    tags: Vec<String>,
    /// Difficulty tier.
    tier: u8,
    /// Min agents.
    min_agents: u32,
    /// Name (lowercased for search).
    name_lower: String,
    /// Description (lowercased for search).
    desc_lower: String,
}

/// Registry of scenario configurations.
///
/// Loads scenarios from TOML files and provides search/filter capabilities.
pub struct ScenarioRegistry {
    scenarios: HashMap<String, ScenarioConfig>,
    index: Vec<IndexEntry>,
}

impl ScenarioRegistry {
    /// Creates a new empty registry.
    pub fn new() -> Self {
        Self {
            scenarios: HashMap::new(),
            index: Vec::new(),
        }
    }

    /// Loads all `.toml` files from a directory into the registry.
    #[instrument(skip_all, fields(path = %path.as_ref().display()))]
    pub fn from_directory(path: impl AsRef<Path>) -> Result<Self, String> {
        let mut registry = Self::new();
        let dir = path.as_ref();

        if !dir.is_dir() {
            return Err(format!("Not a directory: {}", dir.display()));
        }

        let entries =
            std::fs::read_dir(dir).map_err(|e| format!("Failed to read directory: {e}"))?;

        for entry in entries {
            let entry = entry.map_err(|e| format!("Directory entry error: {e}"))?;
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) == Some("toml") {
                match std::fs::read_to_string(&path) {
                    Ok(content) => match ScenarioConfig::from_toml(&content) {
                        Ok(config) => {
                            debug!(id = %config.scenario.id, "Loaded scenario");
                            registry.register(config);
                        }
                        Err(e) => {
                            warn!(path = %path.display(), error = %e, "Failed to parse scenario");
                        }
                    },
                    Err(e) => {
                        warn!(path = %path.display(), error = %e, "Failed to read file");
                    }
                }
            }
        }

        Ok(registry)
    }

    /// Registers a scenario config in the registry.
    ///
    /// If a scenario with the same ID already exists, it is replaced.
    #[instrument(skip(self, config), fields(id = %config.scenario.id))]
    pub fn register(&mut self, config: ScenarioConfig) {
        let entry = IndexEntry {
            id: config.scenario.id.clone(),
            tags: config.scenario.tags.clone(),
            tier: config.scenario.difficulty_tier,
            min_agents: config.scenario.min_agents,
            name_lower: config.scenario.name.to_lowercase(),
            desc_lower: config.scenario.description.to_lowercase(),
        };
        self.index.push(entry);
        self.scenarios.insert(config.scenario.id.clone(), config);
    }

    /// Returns the number of scenarios in the registry.
    pub fn len(&self) -> usize {
        self.scenarios.len()
    }

    /// Returns true if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.scenarios.is_empty()
    }

    /// Gets a scenario by ID.
    pub fn get(&self, id: &str) -> Option<&ScenarioConfig> {
        self.scenarios.get(id)
    }

    /// Returns all scenario IDs.
    pub fn ids(&self) -> Vec<&str> {
        self.scenarios.keys().map(|s| s.as_str()).collect()
    }

    /// Returns scenarios matching the given tier.
    #[instrument(skip(self))]
    pub fn by_tier(&self, tier: u8) -> Vec<&ScenarioConfig> {
        self.index
            .iter()
            .filter(|e| e.tier == tier)
            .filter_map(|e| self.scenarios.get(&e.id))
            .collect()
    }

    /// Returns scenarios matching any of the given tags.
    #[instrument(skip(self))]
    pub fn by_tag(&self, tag: &str) -> Vec<&ScenarioConfig> {
        let tag_lower = tag.to_lowercase();
        self.index
            .iter()
            .filter(|e| e.tags.iter().any(|t| t.to_lowercase() == tag_lower))
            .filter_map(|e| self.scenarios.get(&e.id))
            .collect()
    }

    /// Searches scenarios matching the query.
    ///
    /// Filters are combined with AND logic: all specified filters must match.
    /// An empty query (no filters) returns all scenarios.
    #[instrument(skip(self))]
    pub fn search(&self, query: &ScenarioQuery) -> Vec<&ScenarioConfig> {
        self.index
            .iter()
            .filter(|e| {
                // Tag filter
                if !query.tags.is_empty() {
                    let has_match = query.tags.iter().any(|qt| {
                        let qt_lower = qt.to_lowercase();
                        e.tags.iter().any(|t| t.to_lowercase() == qt_lower)
                    });
                    if !has_match {
                        return false;
                    }
                }

                // Tier range filter
                if let Some((min, max)) = query.tier_range {
                    if e.tier < min || e.tier > max {
                        return false;
                    }
                }

                // Min agents filter
                if let Some(min) = query.min_agents {
                    if e.min_agents < min {
                        return false;
                    }
                }

                // Text search
                if let Some(ref text) = query.text_search {
                    let text_lower = text.to_lowercase();
                    if !e.name_lower.contains(&text_lower) && !e.desc_lower.contains(&text_lower) {
                        return false;
                    }
                }

                true
            })
            .filter_map(|e| self.scenarios.get(&e.id))
            .collect()
    }
}

impl Default for ScenarioRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ScenarioMeta;
    use forge_types::config::ForgeConfig;

    fn make_scenario(id: &str, name: &str, tier: u8, tags: Vec<&str>) -> ScenarioConfig {
        ScenarioConfig {
            scenario: ScenarioMeta {
                id: id.to_string(),
                name: name.to_string(),
                description: format!("Description for {name}"),
                tags: tags.into_iter().map(String::from).collect(),
                difficulty_tier: tier,
                min_agents: 1,
                max_agents: 4,
                author: "test".into(),
                version: "1.0".into(),
            },
            forge: ForgeConfig::default(),
        }
    }

    fn make_populated_registry() -> ScenarioRegistry {
        let mut reg = ScenarioRegistry::new();
        reg.register(make_scenario(
            "patrol",
            "Basic Patrol",
            1,
            vec!["navigation"],
        ));
        reg.register(make_scenario(
            "gather",
            "Resource Gathering",
            2,
            vec!["crafting", "collection"],
        ));
        reg.register(make_scenario("combat", "Arena Combat", 3, vec!["combat"]));
        reg.register(make_scenario(
            "coop",
            "Cooperation Task",
            2,
            vec!["cooperation", "navigation"],
        ));
        reg
    }

    #[test]
    fn test_registry_creation() {
        let reg = ScenarioRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_register_and_get() {
        let mut reg = ScenarioRegistry::new();
        reg.register(make_scenario("test", "Test", 1, vec![]));
        assert_eq!(reg.len(), 1);
        assert!(reg.get("test").is_some());
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn test_by_tier() {
        let reg = make_populated_registry();
        let tier1 = reg.by_tier(1);
        assert_eq!(tier1.len(), 1);
        assert_eq!(tier1[0].scenario.id, "patrol");

        let tier2 = reg.by_tier(2);
        assert_eq!(tier2.len(), 2);
    }

    #[test]
    fn test_by_tag() {
        let reg = make_populated_registry();
        let nav = reg.by_tag("navigation");
        assert_eq!(nav.len(), 2); // patrol + coop

        let combat = reg.by_tag("combat");
        assert_eq!(combat.len(), 1);

        let empty = reg.by_tag("nonexistent");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_search_by_tags() {
        let reg = make_populated_registry();
        let results = reg.search(&ScenarioQuery {
            tags: vec!["crafting".into()],
            ..Default::default()
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].scenario.id, "gather");
    }

    #[test]
    fn test_search_by_tier_range() {
        let reg = make_populated_registry();
        let results = reg.search(&ScenarioQuery {
            tier_range: Some((2, 3)),
            ..Default::default()
        });
        assert_eq!(results.len(), 3); // gather, combat, coop
    }

    #[test]
    fn test_search_by_text() {
        let reg = make_populated_registry();
        let results = reg.search(&ScenarioQuery {
            text_search: Some("patrol".into()),
            ..Default::default()
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].scenario.id, "patrol");
    }

    #[test]
    fn test_search_by_text_description() {
        let reg = make_populated_registry();
        let results = reg.search(&ScenarioQuery {
            text_search: Some("resource".into()),
            ..Default::default()
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].scenario.id, "gather");
    }

    #[test]
    fn test_search_combined() {
        let reg = make_populated_registry();
        let results = reg.search(&ScenarioQuery {
            tags: vec!["navigation".into()],
            tier_range: Some((2, 6)),
            ..Default::default()
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].scenario.id, "coop");
    }

    #[test]
    fn test_search_no_results() {
        let reg = make_populated_registry();
        let results = reg.search(&ScenarioQuery {
            tags: vec!["nonexistent".into()],
            ..Default::default()
        });
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_empty_query_returns_all() {
        let reg = make_populated_registry();
        let results = reg.search(&ScenarioQuery::default());
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_ids() {
        let reg = make_populated_registry();
        let ids = reg.ids();
        assert_eq!(ids.len(), 4);
        assert!(ids.contains(&"patrol"));
        assert!(ids.contains(&"combat"));
    }

    #[test]
    fn test_from_directory_nonexistent() {
        let result = ScenarioRegistry::from_directory("/nonexistent/path");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_directory_with_toml_files() {
        let dir = std::env::temp_dir().join("forge_test_scenarios");
        let _ = std::fs::create_dir_all(&dir);

        // Write a valid scenario TOML
        let toml = r#"
[scenario]
id = "temp_test"
name = "Temp Test"
tags = ["test"]
difficulty_tier = 1
"#;
        std::fs::write(dir.join("test.toml"), toml).unwrap();

        let reg = ScenarioRegistry::from_directory(&dir).unwrap();
        assert_eq!(reg.len(), 1);
        assert!(reg.get("temp_test").is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_case_insensitive_tag_search() {
        let reg = make_populated_registry();
        let results = reg.by_tag("Navigation");
        assert_eq!(results.len(), 2); // case insensitive
    }

    #[test]
    fn test_duplicate_id_replaces() {
        let mut reg = ScenarioRegistry::new();
        reg.register(make_scenario("dup", "First", 1, vec!["a"]));
        reg.register(make_scenario("dup", "Second", 2, vec!["b"]));

        // HashMap replaces, so get returns the latest
        assert_eq!(reg.get("dup").unwrap().scenario.name, "Second");
        // But index has both entries — by_tier finds updated one
        let tier2 = reg.by_tier(2);
        assert!(tier2.iter().any(|s| s.scenario.id == "dup"));
    }

    #[test]
    fn test_search_combined_tags_and_text() {
        let reg = make_populated_registry();
        let results = reg.search(&ScenarioQuery {
            tags: vec!["navigation".into()],
            text_search: Some("cooperation".into()),
            ..Default::default()
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].scenario.id, "coop");
    }

    #[test]
    fn test_search_min_agents_filter() {
        let mut reg = ScenarioRegistry::new();
        let mut s = make_scenario("solo", "Solo", 1, vec![]);
        s.scenario.min_agents = 1;
        reg.register(s);

        let mut s2 = make_scenario("multi", "Multi", 1, vec![]);
        s2.scenario.min_agents = 4;
        reg.register(s2);

        let results = reg.search(&ScenarioQuery {
            min_agents: Some(3),
            ..Default::default()
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].scenario.id, "multi");
    }

    #[test]
    fn test_case_insensitive_text_search() {
        let reg = make_populated_registry();
        let results = reg.search(&ScenarioQuery {
            text_search: Some("PATROL".into()),
            ..Default::default()
        });
        assert_eq!(results.len(), 1);
    }
}
