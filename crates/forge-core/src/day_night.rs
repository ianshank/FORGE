//! Day/night cycle system.
//!
//! Divides simulation time into four phases: dawn, day, dusk, and night.
//! Each phase affects agent vision radius through a modifier, enabling
//! emergent behaviour differences between day and night.

use forge_types::config::WorldConfig;
use forge_types::constants::{
    DAY_PHASE_DAWN, DAY_PHASE_DAY, DAY_PHASE_DUSK, DAY_PHASE_NIGHT, NUM_DAY_PHASES,
    VISION_MODIFIER_DAY, VISION_MODIFIER_NIGHT, VISION_MODIFIER_TWILIGHT,
};
use tracing::{instrument, trace};

/// Dawn phase constant (re-exported from forge-types for convenience).
pub const PHASE_DAWN: u8 = DAY_PHASE_DAWN;
/// Daytime phase constant (re-exported from forge-types for convenience).
pub const PHASE_DAY: u8 = DAY_PHASE_DAY;
/// Dusk phase constant (re-exported from forge-types for convenience).
pub const PHASE_DUSK: u8 = DAY_PHASE_DUSK;
/// Nighttime phase constant (re-exported from forge-types for convenience).
pub const PHASE_NIGHT: u8 = DAY_PHASE_NIGHT;

/// Computes the current day phase (0-3) from the tick and cycle configuration.
///
/// The cycle is divided into 4 equal quarters:
/// - Quarter 0: Dawn
/// - Quarter 1: Day
/// - Quarter 2: Dusk
/// - Quarter 3: Night
///
/// If `day_night_cycle_length` is 0, the cycle is disabled and this
/// always returns `PHASE_DAY`.
#[instrument(skip_all)]
pub fn compute_day_phase(tick: u64, config: &WorldConfig) -> u8 {
    let cycle_length = config.day_night_cycle_length;
    if cycle_length == 0 {
        trace!("day/night cycle disabled, returning DAY");
        return PHASE_DAY;
    }

    let position_in_cycle = tick % (cycle_length as u64);
    let quarter_length = cycle_length as u64 / 4;

    // Avoid division by zero if cycle_length < 4
    if quarter_length == 0 {
        trace!(tick, cycle_length, "cycle length too short, returning DAY");
        return PHASE_DAY;
    }

    let phase = (position_in_cycle / quarter_length).min((NUM_DAY_PHASES - 1) as u64) as u8;

    trace!(tick, cycle_length, phase, "computed day phase");

    phase
}

/// Returns the vision radius modifier for a given day phase.
///
/// - Day: 1.0 (full vision)
/// - Dawn / Dusk: 0.75 (reduced visibility)
/// - Night: 0.5 (limited visibility)
#[instrument]
pub fn vision_modifier(phase: u8) -> f32 {
    match phase {
        PHASE_DAY => VISION_MODIFIER_DAY,
        PHASE_DAWN | PHASE_DUSK => VISION_MODIFIER_TWILIGHT,
        PHASE_NIGHT => VISION_MODIFIER_NIGHT,
        _ => VISION_MODIFIER_DAY, // fallback
    }
}

/// Returns whether the given phase is considered "daytime".
///
/// Dawn and Day are daytime; Dusk and Night are not.
#[instrument]
pub fn is_daytime(phase: u8) -> bool {
    phase == PHASE_DAY || phase == PHASE_DAWN
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::config::WorldConfig;

    fn config_with_cycle(length: u32) -> WorldConfig {
        WorldConfig {
            day_night_cycle_length: length,
            ..WorldConfig::default()
        }
    }

    #[test]
    fn test_day_phase_progression() {
        let config = config_with_cycle(100);

        // Quarter 0 (ticks 0-24): Dawn
        assert_eq!(compute_day_phase(0, &config), PHASE_DAWN);
        assert_eq!(compute_day_phase(12, &config), PHASE_DAWN);
        assert_eq!(compute_day_phase(24, &config), PHASE_DAWN);

        // Quarter 1 (ticks 25-49): Day
        assert_eq!(compute_day_phase(25, &config), PHASE_DAY);
        assert_eq!(compute_day_phase(49, &config), PHASE_DAY);

        // Quarter 2 (ticks 50-74): Dusk
        assert_eq!(compute_day_phase(50, &config), PHASE_DUSK);
        assert_eq!(compute_day_phase(74, &config), PHASE_DUSK);

        // Quarter 3 (ticks 75-99): Night
        assert_eq!(compute_day_phase(75, &config), PHASE_NIGHT);
        assert_eq!(compute_day_phase(99, &config), PHASE_NIGHT);
    }

    #[test]
    fn test_cycle_disabled() {
        let config = config_with_cycle(0);

        assert_eq!(compute_day_phase(0, &config), PHASE_DAY);
        assert_eq!(compute_day_phase(500, &config), PHASE_DAY);
        assert_eq!(compute_day_phase(u64::MAX, &config), PHASE_DAY);
    }

    #[test]
    fn test_vision_modifier_day() {
        assert!((vision_modifier(PHASE_DAY) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_vision_modifier_night() {
        assert!((vision_modifier(PHASE_NIGHT) - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_vision_modifier_dawn_dusk() {
        assert!((vision_modifier(PHASE_DAWN) - 0.75).abs() < f32::EPSILON);
        assert!((vision_modifier(PHASE_DUSK) - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_full_cycle() {
        let config = config_with_cycle(400);
        let mut phases_seen = [false; 4];

        for tick in 0..400 {
            let phase = compute_day_phase(tick, &config);
            assert!(phase <= 3, "phase should be 0-3, got {}", phase);
            phases_seen[phase as usize] = true;
        }

        // All four phases should occur in a full cycle
        assert!(phases_seen.iter().all(|&seen| seen));

        // Cycle should repeat
        for tick in 0..400 {
            assert_eq!(
                compute_day_phase(tick, &config),
                compute_day_phase(tick + 400, &config)
            );
        }
    }

    #[test]
    fn test_is_daytime() {
        assert!(is_daytime(PHASE_DAWN));
        assert!(is_daytime(PHASE_DAY));
        assert!(!is_daytime(PHASE_DUSK));
        assert!(!is_daytime(PHASE_NIGHT));
    }

    #[test]
    fn test_phase_boundary_transition() {
        let config = config_with_cycle(100);

        // Tick 24 → Dawn (last tick of quarter 0)
        assert_eq!(compute_day_phase(24, &config), PHASE_DAWN);
        // Tick 25 → Day (first tick of quarter 1)
        assert_eq!(compute_day_phase(25, &config), PHASE_DAY);
        // Tick 49 → Day (last tick of quarter 1)
        assert_eq!(compute_day_phase(49, &config), PHASE_DAY);
        // Tick 50 → Dusk (first tick of quarter 2)
        assert_eq!(compute_day_phase(50, &config), PHASE_DUSK);
        // Tick 74 → Dusk
        assert_eq!(compute_day_phase(74, &config), PHASE_DUSK);
        // Tick 75 → Night
        assert_eq!(compute_day_phase(75, &config), PHASE_NIGHT);
        // Tick 99 → Night (end of cycle)
        assert_eq!(compute_day_phase(99, &config), PHASE_NIGHT);
        // Tick 100 → Dawn (cycle wraps)
        assert_eq!(compute_day_phase(100, &config), PHASE_DAWN);
    }

    #[test]
    fn test_very_short_cycle() {
        // cycle_length = 3: quarter_length = 0, falls through to PHASE_DAY
        let config = config_with_cycle(3);
        assert_eq!(compute_day_phase(0, &config), PHASE_DAY);
        assert_eq!(compute_day_phase(1, &config), PHASE_DAY);
        assert_eq!(compute_day_phase(2, &config), PHASE_DAY);
    }

    #[test]
    fn test_minimum_valid_cycle() {
        // cycle_length = 4: exactly 1 tick per quarter
        let config = config_with_cycle(4);
        assert_eq!(compute_day_phase(0, &config), PHASE_DAWN);
        assert_eq!(compute_day_phase(1, &config), PHASE_DAY);
        assert_eq!(compute_day_phase(2, &config), PHASE_DUSK);
        assert_eq!(compute_day_phase(3, &config), PHASE_NIGHT);
        // Cycle wraps
        assert_eq!(compute_day_phase(4, &config), PHASE_DAWN);
    }

    #[test]
    fn test_vision_modifier_unknown_phase() {
        // Phase values > 3 should fallback to day modifier
        assert!((vision_modifier(4) - VISION_MODIFIER_DAY).abs() < f32::EPSILON);
        assert!((vision_modifier(255) - VISION_MODIFIER_DAY).abs() < f32::EPSILON);
    }

    #[test]
    fn test_large_tick_values() {
        let config = config_with_cycle(400);
        // Very large tick should not panic and should give valid phase
        let phase = compute_day_phase(u64::MAX - 1, &config);
        assert!(phase <= 3);
    }
}
