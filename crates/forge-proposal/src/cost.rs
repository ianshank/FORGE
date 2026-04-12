//! Cost volume section of an SBIR proposal.
//!
//! Defines labor, materials, travel, and indirect cost structures
//! with configurable rates and automatic total computation.

use serde::{Deserialize, Serialize};

use crate::constants;

/// Complete cost volume for a proposal.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CostVolume {
    /// Labor cost items.
    pub labor: Vec<LaborItem>,
    /// Material cost items.
    pub materials: Vec<MaterialItem>,
    /// Travel cost items.
    pub travel: Vec<TravelItem>,
    /// Subcontract cost items.
    pub subcontracts: Vec<SubcontractItem>,
    /// Indirect cost rates and amounts.
    pub indirect: IndirectCosts,
}

impl CostVolume {
    /// Returns total direct labor cost.
    pub fn total_labor(&self) -> u64 {
        self.labor.iter().map(|l| l.total()).sum()
    }

    /// Returns total materials cost.
    pub fn total_materials(&self) -> u64 {
        self.materials.iter().map(|m| m.cost).sum()
    }

    /// Returns total travel cost.
    pub fn total_travel(&self) -> u64 {
        self.travel.iter().map(|t| t.cost).sum()
    }

    /// Returns total subcontract cost.
    pub fn total_subcontracts(&self) -> u64 {
        self.subcontracts.iter().map(|s| s.cost).sum()
    }

    /// Returns total direct costs (labor + materials + travel + subcontracts).
    pub fn total_direct(&self) -> u64 {
        self.total_labor()
            + self.total_materials()
            + self.total_travel()
            + self.total_subcontracts()
    }

    /// Returns total indirect costs based on the configured rates and the direct cost base.
    pub fn total_indirect(&self) -> u64 {
        let base = self.total_labor(); // Indirect rates typically apply to labor
        let overhead = (base as f64 * self.indirect.overhead_rate as f64) as u64;
        let ga = ((base + overhead) as f64 * self.indirect.ga_rate as f64) as u64;
        overhead + ga
    }

    /// Returns the total proposal cost (direct + indirect + profit).
    pub fn total_cost(&self) -> u64 {
        let subtotal = self.total_direct() + self.total_indirect();
        let profit = (subtotal as f64 * self.indirect.profit_rate as f64) as u64;
        subtotal + profit
    }

    /// Returns the subcontract percentage of total cost.
    pub fn subcontract_percentage(&self) -> f32 {
        let total = self.total_cost();
        if total == 0 {
            return 0.0;
        }
        (self.total_subcontracts() as f32 / total as f32) * 100.0
    }

    /// Estimates page count for the cost volume.
    pub fn estimated_pages(&self) -> f32 {
        // Cost volumes are typically 1-2 pages of tables
        let line_count = self.labor.len()
            + self.materials.len()
            + self.travel.len()
            + self.subcontracts.len()
            + 5; // header rows
        line_count as f32 / 40.0 // ~40 table rows per page
    }

    /// Renders this cost volume to Markdown.
    pub fn render_markdown(&self, heading_level: u8) -> String {
        let h = "#".repeat(heading_level as usize);
        let mut out = String::new();
        out.push_str(&format!("{h} Cost Volume\n\n"));

        // Labor
        if !self.labor.is_empty() {
            out.push_str("**Labor:**\n\n");
            out.push_str("| Category | Hours | Rate ($/hr) | Total |\n");
            out.push_str("|----------|-------|-------------|-------|\n");
            for item in &self.labor {
                out.push_str(&format!(
                    "| {} | {} | {} | ${} |\n",
                    item.category,
                    item.hours,
                    item.hourly_rate,
                    item.total(),
                ));
            }
            out.push_str(&format!(
                "| **Total Labor** | | | **${}** |\n\n",
                self.total_labor()
            ));
        }

        // Materials
        if !self.materials.is_empty() {
            out.push_str("**Materials:**\n\n");
            for item in &self.materials {
                out.push_str(&format!("- {}: ${}\n", item.description, item.cost));
            }
            out.push_str(&format!(
                "\n**Total Materials:** ${}\n\n",
                self.total_materials()
            ));
        }

        // Travel
        if !self.travel.is_empty() {
            out.push_str("**Travel:**\n\n");
            for item in &self.travel {
                out.push_str(&format!("- {}: ${}\n", item.description, item.cost));
            }
            out.push_str(&format!("\n**Total Travel:** ${}\n\n", self.total_travel()));
        }

        // Subcontracts
        if !self.subcontracts.is_empty() {
            out.push_str("**Subcontracts:**\n\n");
            for item in &self.subcontracts {
                out.push_str(&format!(
                    "- {} ({}): ${}\n",
                    item.organization, item.description, item.cost
                ));
            }
            out.push_str(&format!(
                "\n**Total Subcontracts:** ${}\n\n",
                self.total_subcontracts(),
            ));
        }

        // Indirect costs
        out.push_str(&format!(
            "**Indirect Costs:** Overhead {:.0}%, G&A {:.0}%, Profit {:.0}%\n\n",
            self.indirect.overhead_rate * 100.0,
            self.indirect.ga_rate * 100.0,
            self.indirect.profit_rate * 100.0,
        ));
        out.push_str(&format!(
            "**Total Proposed Cost:** ${}\n\n",
            self.total_cost()
        ));
        out
    }
}

/// A single labor cost item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaborItem {
    /// Labor category (e.g., "Principal Investigator", "Research Engineer").
    pub category: String,
    /// Number of hours.
    pub hours: u32,
    /// Hourly rate in USD.
    pub hourly_rate: u32,
}

impl LaborItem {
    /// Returns the total cost for this labor item.
    pub fn total(&self) -> u64 {
        self.hours as u64 * self.hourly_rate as u64
    }
}

/// A single material cost item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialItem {
    /// Description of the material.
    pub description: String,
    /// Cost in USD.
    pub cost: u64,
}

/// A single travel cost item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TravelItem {
    /// Description of the travel.
    pub description: String,
    /// Cost in USD.
    pub cost: u64,
}

/// A single subcontract cost item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubcontractItem {
    /// Subcontractor organization name.
    pub organization: String,
    /// Description of work.
    pub description: String,
    /// Cost in USD.
    pub cost: u64,
}

/// Indirect cost rates.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IndirectCosts {
    /// Overhead rate (fraction, e.g., 0.40 = 40%).
    pub overhead_rate: f32,
    /// General & Administrative rate (fraction).
    pub ga_rate: f32,
    /// Profit/fee rate (fraction).
    pub profit_rate: f32,
}

impl Default for IndirectCosts {
    fn default() -> Self {
        Self {
            overhead_rate: constants::DEFAULT_OVERHEAD_RATE,
            ga_rate: constants::DEFAULT_GA_RATE,
            profit_rate: constants::DEFAULT_PROFIT_RATE_MAX,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_volume_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(CostVolume);
    }

    #[test]
    fn test_cost_volume_defaults_valid() {
        forge_types::assert_config_defaults_valid!(CostVolume);
    }

    #[test]
    fn test_indirect_costs_serde_roundtrip() {
        forge_types::assert_config_serde_roundtrip!(IndirectCosts);
    }

    #[test]
    fn test_labor_item_total() {
        let item = LaborItem {
            category: "PI".to_string(),
            hours: 100,
            hourly_rate: 150,
        };
        assert_eq!(item.total(), 15_000);
    }

    #[test]
    fn test_empty_cost_volume_totals() {
        let vol = CostVolume::default();
        assert_eq!(vol.total_labor(), 0);
        assert_eq!(vol.total_materials(), 0);
        assert_eq!(vol.total_travel(), 0);
        assert_eq!(vol.total_subcontracts(), 0);
        assert_eq!(vol.total_direct(), 0);
        assert_eq!(vol.total_indirect(), 0);
        assert_eq!(vol.total_cost(), 0);
    }

    #[test]
    fn test_cost_volume_with_labor() {
        let vol = CostVolume {
            labor: vec![
                LaborItem {
                    category: "PI".to_string(),
                    hours: 1000,
                    hourly_rate: 100,
                },
                LaborItem {
                    category: "Engineer".to_string(),
                    hours: 500,
                    hourly_rate: 80,
                },
            ],
            ..Default::default()
        };
        assert_eq!(vol.total_labor(), 140_000);
        assert!(vol.total_cost() > vol.total_labor()); // indirect costs add up
    }

    #[test]
    fn test_subcontract_percentage() {
        let vol = CostVolume {
            labor: vec![LaborItem {
                category: "PI".to_string(),
                hours: 1000,
                hourly_rate: 100,
            }],
            subcontracts: vec![SubcontractItem {
                organization: "Sub Inc".to_string(),
                description: "Analysis".to_string(),
                cost: 50_000,
            }],
            ..Default::default()
        };
        let pct = vol.subcontract_percentage();
        assert!(pct > 0.0 && pct < 100.0);
    }

    #[test]
    fn test_subcontract_percentage_zero_cost() {
        let vol = CostVolume::default();
        assert!((vol.subcontract_percentage()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_indirect_cost_defaults() {
        let indirect = IndirectCosts::default();
        assert!((indirect.overhead_rate - constants::DEFAULT_OVERHEAD_RATE).abs() < f32::EPSILON);
        assert!((indirect.ga_rate - constants::DEFAULT_GA_RATE).abs() < f32::EPSILON);
        assert!((indirect.profit_rate - constants::DEFAULT_PROFIT_RATE_MAX).abs() < f32::EPSILON);
    }

    #[test]
    fn test_render_cost_volume_empty() {
        let vol = CostVolume::default();
        let md = vol.render_markdown(1);
        assert!(md.contains("# Cost Volume"));
        assert!(md.contains("Total Proposed Cost"));
    }

    #[test]
    fn test_render_cost_volume_with_data() {
        let vol = CostVolume {
            labor: vec![LaborItem {
                category: "PI".to_string(),
                hours: 500,
                hourly_rate: 100,
            }],
            materials: vec![MaterialItem {
                description: "GPU Server".to_string(),
                cost: 5_000,
            }],
            travel: vec![TravelItem {
                description: "Conference".to_string(),
                cost: 2_000,
            }],
            subcontracts: vec![SubcontractItem {
                organization: "Uni".to_string(),
                description: "Research".to_string(),
                cost: 30_000,
            }],
            ..Default::default()
        };
        let md = vol.render_markdown(2);
        assert!(md.contains("## Cost Volume"));
        assert!(md.contains("PI"));
        assert!(md.contains("GPU Server"));
        assert!(md.contains("Conference"));
        assert!(md.contains("Uni"));
        assert!(md.contains("Total Labor"));
        assert!(md.contains("Total Proposed Cost"));
    }

    #[test]
    fn test_estimated_pages_scales_with_items() {
        let small = CostVolume::default();
        let big = CostVolume {
            labor: (0..10)
                .map(|i| LaborItem {
                    category: format!("Cat {i}"),
                    hours: 100,
                    hourly_rate: 100,
                })
                .collect(),
            ..Default::default()
        };
        assert!(big.estimated_pages() > small.estimated_pages());
    }
}
