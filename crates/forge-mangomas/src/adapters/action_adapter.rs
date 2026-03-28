//! Bidirectional action space mapping between FORGE discrete and MangoMAS continuous.
//!
//! FORGE uses 40+ discrete actions; MangoMAS car uses 2D continuous `[steering, throttle]`,
//! MangoMAS drone uses 4D continuous `[vx, vy, vz, yaw_rate]`. The adapter discretizes
//! the continuous space into a grid of bins and maps each bin to the semantically
//! closest FORGE action.

use forge_types::action::Action;
use forge_types::grid::Direction;
use tracing::instrument;

use crate::config::{ActionAdapterConfig, Platform};
use crate::error::{MangoMasError, MangoMasResult};

/// Trait for bidirectional action space adaptation.
///
/// Implementations convert between MangoMAS continuous action vectors
/// and FORGE discrete actions.
pub trait ActionAdapter: Send + Sync {
    /// Maps a continuous action vector to a FORGE discrete action.
    fn continuous_to_discrete(&self, continuous: &[f32]) -> MangoMasResult<Action>;

    /// Maps a FORGE discrete action to a continuous action vector.
    fn discrete_to_continuous(&self, action: &Action) -> MangoMasResult<Vec<f32>>;

    /// Returns the dimensionality of the continuous action space.
    fn continuous_dims(&self) -> usize;
}

/// Maps FORGE discrete actions to MangoMAS BDI-relevant semantic categories.
///
/// This is used alongside the action adapter to provide intention-level
/// understanding of actions for the BDI pre-training pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionCategory {
    /// Navigation movement.
    Navigate,
    /// Resource gathering.
    Gather,
    /// Crafting/planning.
    Plan,
    /// Object manipulation.
    Manipulate,
    /// Communication.
    Cooperate,
    /// Combat/evasion.
    Evade,
    /// Scanning/tracking.
    Track,
    /// Idle/hover/noop.
    Idle,
}

impl ActionCategory {
    /// Returns the BDI intention class index (0-7).
    pub fn as_intention_index(&self) -> u8 {
        match self {
            Self::Navigate => 0,
            Self::Gather => 1,
            Self::Plan => 2,
            Self::Manipulate => 3,
            Self::Cooperate => 4,
            Self::Evade => 5,
            Self::Track => 6,
            Self::Idle => 7,
        }
    }

    /// Classifies a FORGE action into a semantic category.
    #[instrument(skip_all)]
    pub fn from_action(action: &Action) -> Self {
        match action {
            Action::Move(_) => Self::Navigate,
            Action::PickUp | Action::Drop(_) => Self::Gather,
            Action::Craft(_) => Self::Plan,
            Action::Push(_) | Action::Use(_) | Action::Interact => Self::Manipulate,
            Action::Communicate(_) => Self::Cooperate,
            Action::Scan(_) => Self::Track,
            Action::Ascend | Action::Descend | Action::TakeOff | Action::Land => Self::Navigate,
            Action::DropPayload(_) => Self::Gather,
            Action::Noop | Action::Hover => Self::Idle,
            _ => Self::Idle, // Catch future non-exhaustive variants
        }
    }
}

/// Grid-based discretization adapter for continuous action spaces.
///
/// Divides each continuous axis into `bins_per_axis` bins and maps
/// each bin combination to the semantically closest FORGE action.
pub struct DiscreteGridAdapter {
    config: ActionAdapterConfig,
    /// Precomputed bin width for each axis.
    bin_width: f32,
}

impl DiscreteGridAdapter {
    /// Creates a new adapter from configuration.
    #[instrument(skip_all)]
    pub fn new(config: ActionAdapterConfig) -> MangoMasResult<Self> {
        if config.bins_per_axis == 0 {
            return Err(MangoMasError::ActionAdapter(
                "bins_per_axis must be > 0".to_string(),
            ));
        }
        if config.action_range_max <= config.action_range_min {
            return Err(MangoMasError::ActionAdapter(
                "action_range_max must be > action_range_min".to_string(),
            ));
        }
        let bin_width =
            (config.action_range_max - config.action_range_min) / config.bins_per_axis as f32;
        Ok(Self { config, bin_width })
    }

    /// Quantizes a single continuous value to a bin index.
    fn quantize(&self, value: f32) -> u32 {
        let clamped = value.clamp(self.config.action_range_min, self.config.action_range_max);
        let normalized = (clamped - self.config.action_range_min) / self.bin_width;
        let bin = normalized.floor() as u32;
        bin.min(self.config.bins_per_axis - 1)
    }

    /// Returns the center value of a bin.
    fn bin_center(&self, bin: u32) -> f32 {
        self.config.action_range_min + (bin as f32 + 0.5) * self.bin_width
    }

    /// Maps 2D car bins (steering, throttle) to a FORGE action.
    fn car_bins_to_action(&self, steer_bin: u32, throttle_bin: u32) -> Action {
        let mid = self.config.bins_per_axis / 2;
        let throttle_high = throttle_bin > mid;
        let throttle_low = throttle_bin < mid;
        let steer_left = steer_bin < mid;
        let steer_right = steer_bin > mid;

        if throttle_high {
            if steer_left {
                Action::Move(Direction::Left)
            } else if steer_right {
                Action::Move(Direction::Right)
            } else {
                Action::Move(Direction::Up)
            }
        } else if throttle_low {
            Action::Move(Direction::Down)
        } else if steer_left {
            Action::Move(Direction::Left)
        } else if steer_right {
            Action::Move(Direction::Right)
        } else {
            Action::Noop
        }
    }

    /// Maps 4D drone bins (vx, vy, vz, yaw) to a FORGE action.
    fn drone_bins_to_action(&self, vx: u32, vy: u32, vz: u32, _yaw: u32) -> Action {
        let mid = self.config.bins_per_axis / 2;
        let threshold = self.config.bins_per_axis / 3;

        // Vertical axis takes priority for altitude control
        if vz > mid + threshold {
            return Action::Ascend;
        }
        if vz < mid.saturating_sub(threshold) {
            return Action::Descend;
        }

        // Horizontal movement
        let dx = vx as i32 - mid as i32;
        let dy = vy as i32 - mid as i32;

        if dx.abs() > dy.abs() {
            if dx > 0 {
                Action::Move(Direction::Right)
            } else {
                Action::Move(Direction::Left)
            }
        } else if dy.abs() > 0 {
            if dy > 0 {
                Action::Move(Direction::Down)
            } else {
                Action::Move(Direction::Up)
            }
        } else {
            Action::Hover
        }
    }
}

impl ActionAdapter for DiscreteGridAdapter {
    #[instrument(skip_all)]
    fn continuous_to_discrete(&self, continuous: &[f32]) -> MangoMasResult<Action> {
        let expected = self.continuous_dims();
        if continuous.len() != expected {
            return Err(MangoMasError::ActionAdapter(format!(
                "expected {} dims, got {}",
                expected,
                continuous.len()
            )));
        }

        match self.config.platform {
            Platform::Car => {
                let steer_bin = self.quantize(continuous[0]);
                let throttle_bin = self.quantize(continuous[1]);
                Ok(self.car_bins_to_action(steer_bin, throttle_bin))
            }
            Platform::Drone => {
                let vx = self.quantize(continuous[0]);
                let vy = self.quantize(continuous[1]);
                let vz = self.quantize(continuous[2]);
                let yaw = self.quantize(continuous[3]);
                Ok(self.drone_bins_to_action(vx, vy, vz, yaw))
            }
        }
    }

    #[instrument(skip_all)]
    fn discrete_to_continuous(&self, action: &Action) -> MangoMasResult<Vec<f32>> {
        let mid = self.config.bins_per_axis / 2;
        let high = self.config.bins_per_axis - 1;
        let center = |bin: u32| self.bin_center(bin);

        match self.config.platform {
            Platform::Car => {
                let (steer, throttle) = match action {
                    Action::Move(Direction::Up) => (mid, high),
                    Action::Move(Direction::Down) => (mid, 0),
                    Action::Move(Direction::Left) => (0, mid),
                    Action::Move(Direction::Right) => (high, mid),
                    Action::Noop | Action::Hover => (mid, mid),
                    _ => (mid, mid),
                };
                Ok(vec![center(steer), center(throttle)])
            }
            Platform::Drone => {
                let (vx, vy, vz, yaw) = match action {
                    Action::Move(Direction::Up) => (mid, 0, mid, mid),
                    Action::Move(Direction::Down) => (mid, high, mid, mid),
                    Action::Move(Direction::Left) => (0, mid, mid, mid),
                    Action::Move(Direction::Right) => (high, mid, mid, mid),
                    Action::Ascend => (mid, mid, high, mid),
                    Action::Descend => (mid, mid, 0, mid),
                    Action::Hover | Action::Noop => (mid, mid, mid, mid),
                    Action::TakeOff => (mid, mid, high, mid),
                    Action::Land => (mid, mid, 0, mid),
                    _ => (mid, mid, mid, mid),
                };
                Ok(vec![center(vx), center(vy), center(vz), center(yaw)])
            }
        }
    }

    fn continuous_dims(&self) -> usize {
        match self.config.platform {
            Platform::Car => 2,
            Platform::Drone => 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn car_adapter() -> DiscreteGridAdapter {
        DiscreteGridAdapter::new(ActionAdapterConfig {
            platform: Platform::Car,
            ..ActionAdapterConfig::default()
        })
        .unwrap()
    }

    fn drone_adapter() -> DiscreteGridAdapter {
        DiscreteGridAdapter::new(ActionAdapterConfig {
            platform: Platform::Drone,
            ..ActionAdapterConfig::default()
        })
        .unwrap()
    }

    #[test]
    fn test_car_continuous_dims() {
        assert_eq!(car_adapter().continuous_dims(), 2);
    }

    #[test]
    fn test_drone_continuous_dims() {
        assert_eq!(drone_adapter().continuous_dims(), 4);
    }

    #[test]
    fn test_car_center_maps_to_noop() {
        let adapter = car_adapter();
        let action = adapter.continuous_to_discrete(&[0.0, 0.0]).unwrap();
        assert_eq!(action, Action::Noop);
    }

    #[test]
    fn test_car_forward() {
        let adapter = car_adapter();
        let action = adapter.continuous_to_discrete(&[0.0, 0.9]).unwrap();
        assert_eq!(action, Action::Move(Direction::Up));
    }

    #[test]
    fn test_car_reverse() {
        let adapter = car_adapter();
        let action = adapter.continuous_to_discrete(&[0.0, -0.9]).unwrap();
        assert_eq!(action, Action::Move(Direction::Down));
    }

    #[test]
    fn test_car_wrong_dims() {
        let adapter = car_adapter();
        let result = adapter.continuous_to_discrete(&[0.0, 0.0, 0.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_drone_ascend() {
        let adapter = drone_adapter();
        let action = adapter
            .continuous_to_discrete(&[0.0, 0.0, 0.9, 0.0])
            .unwrap();
        assert_eq!(action, Action::Ascend);
    }

    #[test]
    fn test_drone_descend() {
        let adapter = drone_adapter();
        let action = adapter
            .continuous_to_discrete(&[0.0, 0.0, -0.9, 0.0])
            .unwrap();
        assert_eq!(action, Action::Descend);
    }

    #[test]
    fn test_drone_hover_at_center() {
        let adapter = drone_adapter();
        let action = adapter
            .continuous_to_discrete(&[0.0, 0.0, 0.0, 0.0])
            .unwrap();
        assert_eq!(action, Action::Hover);
    }

    #[test]
    fn test_roundtrip_noop() {
        let adapter = car_adapter();
        let continuous = adapter.discrete_to_continuous(&Action::Noop).unwrap();
        assert_eq!(continuous.len(), 2);
        let action_back = adapter.continuous_to_discrete(&continuous).unwrap();
        assert_eq!(action_back, Action::Noop);
    }

    #[test]
    fn test_roundtrip_drone_hover() {
        let adapter = drone_adapter();
        let continuous = adapter.discrete_to_continuous(&Action::Hover).unwrap();
        assert_eq!(continuous.len(), 4);
        let action_back = adapter.continuous_to_discrete(&continuous).unwrap();
        assert_eq!(action_back, Action::Hover);
    }

    #[test]
    fn test_invalid_bins_zero() {
        let result = DiscreteGridAdapter::new(ActionAdapterConfig {
            bins_per_axis: 0,
            ..ActionAdapterConfig::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_range() {
        let result = DiscreteGridAdapter::new(ActionAdapterConfig {
            action_range_min: 1.0,
            action_range_max: -1.0,
            ..ActionAdapterConfig::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_action_category_from_action() {
        assert_eq!(
            ActionCategory::from_action(&Action::Move(Direction::Up)),
            ActionCategory::Navigate
        );
        assert_eq!(
            ActionCategory::from_action(&Action::PickUp),
            ActionCategory::Gather
        );
        assert_eq!(
            ActionCategory::from_action(&Action::Craft(0)),
            ActionCategory::Plan
        );
        assert_eq!(
            ActionCategory::from_action(&Action::Push(Direction::Left)),
            ActionCategory::Manipulate
        );
        assert_eq!(
            ActionCategory::from_action(&Action::Communicate(0)),
            ActionCategory::Cooperate
        );
        assert_eq!(
            ActionCategory::from_action(&Action::Scan(Direction::Up)),
            ActionCategory::Track
        );
        assert_eq!(
            ActionCategory::from_action(&Action::Noop),
            ActionCategory::Idle
        );
        assert_eq!(
            ActionCategory::from_action(&Action::Hover),
            ActionCategory::Idle
        );
        assert_eq!(
            ActionCategory::from_action(&Action::Ascend),
            ActionCategory::Navigate
        );
    }

    #[test]
    fn test_action_category_intention_indices_unique() {
        let categories = [
            ActionCategory::Navigate,
            ActionCategory::Gather,
            ActionCategory::Plan,
            ActionCategory::Manipulate,
            ActionCategory::Cooperate,
            ActionCategory::Evade,
            ActionCategory::Track,
            ActionCategory::Idle,
        ];
        let indices: Vec<u8> = categories.iter().map(|c| c.as_intention_index()).collect();
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(idx, i as u8, "intention indices must be 0..7 in order");
        }
    }

    #[test]
    fn test_quantize_clamps() {
        let adapter = car_adapter();
        // Values beyond range should be clamped
        let bin_low = adapter.quantize(-100.0);
        let bin_high = adapter.quantize(100.0);
        assert_eq!(bin_low, 0);
        assert_eq!(bin_high, adapter.config.bins_per_axis - 1);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// All continuous inputs produce valid actions.
        #[test]
        fn car_always_produces_valid_action(
            steer in -2.0f32..2.0,
            throttle in -2.0f32..2.0,
        ) {
            let adapter = DiscreteGridAdapter::new(ActionAdapterConfig {
                platform: Platform::Car,
                ..ActionAdapterConfig::default()
            }).unwrap();
            let result = adapter.continuous_to_discrete(&[steer, throttle]);
            prop_assert!(result.is_ok());
        }

        /// All continuous drone inputs produce valid actions.
        #[test]
        fn drone_always_produces_valid_action(
            vx in -2.0f32..2.0,
            vy in -2.0f32..2.0,
            vz in -2.0f32..2.0,
            yaw in -2.0f32..2.0,
        ) {
            let adapter = DiscreteGridAdapter::new(ActionAdapterConfig {
                platform: Platform::Drone,
                ..ActionAdapterConfig::default()
            }).unwrap();
            let result = adapter.continuous_to_discrete(&[vx, vy, vz, yaw]);
            prop_assert!(result.is_ok());
        }

        /// discrete_to_continuous always returns correct dimensionality.
        #[test]
        fn discrete_to_continuous_correct_dims(action_id in 0u32..40) {
            let adapter = DiscreteGridAdapter::new(ActionAdapterConfig {
                platform: Platform::Car,
                ..ActionAdapterConfig::default()
            }).unwrap();
            if let Some(action) = Action::from_discrete(action_id, 16, false) {
                let continuous = adapter.discrete_to_continuous(&action).unwrap();
                prop_assert_eq!(continuous.len(), 2);
            }
        }
    }
}
