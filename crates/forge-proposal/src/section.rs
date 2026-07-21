//! Composable proposal section model.
//!
//! [`SectionComposition`] follows the recursive tree pattern established by
//! `TaskComposition` in `forge-task`. This allows agency-adaptive proposal
//! structures where different agencies require different section arrangements.

use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Content for a leaf section in the composition tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionContent {
    /// Section identifier (e.g., "problem", "approach").
    pub id: String,
    /// Human-readable section title.
    pub title: String,
    /// Section body text.
    pub body: String,
    /// Estimated page count.
    pub estimated_pages: f32,
}

/// Composable proposal sections, following the forge-task composition pattern.
///
/// Allows expressing complex section structures that can adapt to different
/// agency requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SectionComposition {
    /// A single leaf section with content.
    Leaf(SectionContent),
    /// All sub-sections must be present, rendered in order.
    Sequence(Vec<SectionComposition>),
    /// Exactly one of the sub-sections is selected.
    OneOf(Vec<SectionComposition>),
    /// An optional section that may be omitted.
    Optional(Box<SectionComposition>),
    /// A section with agency-imposed constraints.
    Constrained {
        /// The section being constrained.
        section: Box<SectionComposition>,
        /// Maximum pages for this section.
        max_pages: Option<u32>,
        /// Whether this section is required by the agency.
        required: bool,
    },
}

impl SectionComposition {
    /// Creates a new leaf section.
    pub fn leaf(
        id: impl Into<String>,
        title: impl Into<String>,
        body: impl Into<String>,
        estimated_pages: f32,
    ) -> Self {
        Self::Leaf(SectionContent {
            id: id.into(),
            title: title.into(),
            body: body.into(),
            estimated_pages,
        })
    }

    /// Creates a sequence of sections.
    pub fn sequence(sections: Vec<SectionComposition>) -> Self {
        Self::Sequence(sections)
    }

    /// Creates a constrained section.
    pub fn constrained(
        section: SectionComposition,
        max_pages: Option<u32>,
        required: bool,
    ) -> Self {
        Self::Constrained {
            section: Box::new(section),
            max_pages,
            required,
        }
    }

    /// Creates an optional section.
    pub fn optional(section: SectionComposition) -> Self {
        Self::Optional(Box::new(section))
    }

    /// Returns the total estimated pages for this composition tree.
    #[instrument(skip_all)]
    pub fn total_estimated_pages(&self) -> f32 {
        match self {
            Self::Leaf(content) => content.estimated_pages,
            Self::Sequence(sections) => sections.iter().map(|s| s.total_estimated_pages()).sum(),
            Self::OneOf(sections) => {
                // Estimate based on the largest option
                sections
                    .iter()
                    .map(|s| s.total_estimated_pages())
                    .fold(0.0_f32, f32::max)
            }
            Self::Optional(section) => section.total_estimated_pages(),
            Self::Constrained { section, .. } => section.total_estimated_pages(),
        }
    }

    /// Returns all leaf section IDs in this composition tree.
    pub fn leaf_ids(&self) -> Vec<&str> {
        match self {
            Self::Leaf(content) => vec![&content.id],
            Self::Sequence(sections) | Self::OneOf(sections) => {
                sections.iter().flat_map(|s| s.leaf_ids()).collect()
            }
            Self::Optional(section) | Self::Constrained { section, .. } => section.leaf_ids(),
        }
    }

    /// Returns the number of leaf sections in this composition tree.
    pub fn leaf_count(&self) -> usize {
        match self {
            Self::Leaf(_) => 1,
            Self::Sequence(sections) | Self::OneOf(sections) => {
                sections.iter().map(|s| s.leaf_count()).sum()
            }
            Self::Optional(section) | Self::Constrained { section, .. } => section.leaf_count(),
        }
    }

    /// Returns the depth of this composition tree.
    pub fn depth(&self) -> usize {
        match self {
            Self::Leaf(_) => 1,
            Self::Sequence(sections) | Self::OneOf(sections) => {
                1 + sections.iter().map(|s| s.depth()).max().unwrap_or(0)
            }
            Self::Optional(section) | Self::Constrained { section, .. } => 1 + section.depth(),
        }
    }

    /// Collects all page constraints from the tree.
    #[instrument(skip_all)]
    pub fn page_constraints(&self) -> Vec<(&str, u32)> {
        match self {
            Self::Leaf(_) => vec![],
            Self::Sequence(sections) | Self::OneOf(sections) => {
                sections.iter().flat_map(|s| s.page_constraints()).collect()
            }
            Self::Optional(section) => section.page_constraints(),
            Self::Constrained {
                section,
                max_pages: Some(limit),
                ..
            } => {
                let mut constraints: Vec<(&str, u32)> = section.page_constraints();
                for id in section.leaf_ids() {
                    constraints.push((id, *limit));
                }
                constraints
            }
            Self::Constrained {
                section,
                max_pages: None,
                ..
            } => section.page_constraints(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_leaf(id: &str, pages: f32) -> SectionComposition {
        SectionComposition::leaf(id, id, "", pages)
    }

    #[test]
    fn test_leaf_estimated_pages() {
        let leaf = make_leaf("test", 2.5);
        assert!((leaf.total_estimated_pages() - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sequence_estimated_pages() {
        let seq = SectionComposition::sequence(vec![make_leaf("a", 1.0), make_leaf("b", 2.0)]);
        assert!((seq.total_estimated_pages() - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_one_of_estimated_pages() {
        let one_of = SectionComposition::OneOf(vec![make_leaf("a", 1.0), make_leaf("b", 3.0)]);
        assert!((one_of.total_estimated_pages() - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_optional_estimated_pages() {
        let opt = SectionComposition::optional(make_leaf("a", 2.0));
        assert!((opt.total_estimated_pages() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_constrained_estimated_pages() {
        let c = SectionComposition::constrained(make_leaf("a", 5.0), Some(10), true);
        assert!((c.total_estimated_pages() - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_leaf_ids() {
        let seq = SectionComposition::sequence(vec![
            make_leaf("a", 1.0),
            make_leaf("b", 1.0),
            SectionComposition::optional(make_leaf("c", 1.0)),
        ]);
        let ids = seq.leaf_ids();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_leaf_count() {
        let seq = SectionComposition::sequence(vec![
            make_leaf("a", 1.0),
            SectionComposition::OneOf(vec![make_leaf("b", 1.0), make_leaf("c", 1.0)]),
        ]);
        assert_eq!(seq.leaf_count(), 3);
    }

    #[test]
    fn test_depth_leaf() {
        assert_eq!(make_leaf("a", 1.0).depth(), 1);
    }

    #[test]
    fn test_depth_nested() {
        let nested = SectionComposition::sequence(vec![SectionComposition::constrained(
            SectionComposition::sequence(vec![make_leaf("a", 1.0)]),
            Some(10),
            true,
        )]);
        assert_eq!(nested.depth(), 4);
    }

    #[test]
    fn test_page_constraints() {
        let tree = SectionComposition::constrained(make_leaf("tech", 5.0), Some(10), true);
        let constraints = tree.page_constraints();
        assert_eq!(constraints.len(), 1);
        assert_eq!(constraints[0], ("tech", 10));
    }

    #[test]
    fn test_page_constraints_empty() {
        let leaf = make_leaf("a", 1.0);
        assert!(leaf.page_constraints().is_empty());
    }

    #[test]
    fn test_serde_roundtrip_leaf() {
        let leaf = make_leaf("test", 1.0);
        let json = serde_json::to_string(&leaf).unwrap();
        let deser: SectionComposition = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.leaf_count(), 1);
    }

    #[test]
    fn test_serde_roundtrip_complex() {
        let tree = SectionComposition::sequence(vec![
            SectionComposition::constrained(make_leaf("a", 1.0), Some(5), true),
            SectionComposition::optional(make_leaf("b", 2.0)),
        ]);
        let json = serde_json::to_string(&tree).unwrap();
        let deser: SectionComposition = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.leaf_count(), 2);
        assert!((deser.total_estimated_pages() - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_one_of_page_constraints() {
        let one_of = SectionComposition::OneOf(vec![
            SectionComposition::constrained(make_leaf("a", 1.0), Some(5), true),
            make_leaf("b", 2.0),
        ]);
        let constraints = one_of.page_constraints();
        assert_eq!(constraints.len(), 1);
        assert_eq!(constraints[0], ("a", 5));
    }

    #[test]
    fn test_constrained_no_page_limit() {
        let c = SectionComposition::constrained(make_leaf("x", 3.0), None, true);
        let constraints = c.page_constraints();
        assert!(constraints.is_empty());
    }

    #[test]
    fn test_optional_page_constraints() {
        let opt = SectionComposition::optional(SectionComposition::constrained(
            make_leaf("opt", 2.0),
            Some(8),
            false,
        ));
        let constraints = opt.page_constraints();
        assert_eq!(constraints.len(), 1);
    }

    #[test]
    fn test_leaf_body_preserved() {
        let leaf = SectionComposition::leaf("id", "Title", "Body content here.", 1.0);
        if let SectionComposition::Leaf(content) = leaf {
            assert_eq!(content.body, "Body content here.");
            assert_eq!(content.title, "Title");
        } else {
            panic!("expected Leaf variant");
        }
    }

    #[test]
    fn test_empty_sequence() {
        let seq = SectionComposition::sequence(vec![]);
        assert_eq!(seq.leaf_count(), 0);
        assert!((seq.total_estimated_pages()).abs() < f32::EPSILON);
        assert!(seq.leaf_ids().is_empty());
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        prop_compose! {
            fn arb_leaf()(
                pages in 0.0f32..100.0,
                id_num in 0u32..1000,
            ) -> SectionComposition {
                SectionComposition::leaf(
                    format!("sec-{id_num}"),
                    format!("Section {id_num}"),
                    "Body text.",
                    pages,
                )
            }
        }

        proptest! {
            #[test]
            fn prop_leaf_has_one_count(leaf in arb_leaf()) {
                prop_assert_eq!(leaf.leaf_count(), 1);
            }

            #[test]
            fn prop_leaf_depth_is_one(leaf in arb_leaf()) {
                prop_assert_eq!(leaf.depth(), 1);
            }

            #[test]
            fn prop_leaf_pages_non_negative(pages in 0.0f32..1000.0) {
                let leaf = SectionComposition::leaf("test", "Test", "", pages);
                prop_assert!(leaf.total_estimated_pages() >= 0.0);
            }

            #[test]
            fn prop_sequence_pages_is_sum(
                p1 in 0.0f32..50.0,
                p2 in 0.0f32..50.0,
                p3 in 0.0f32..50.0,
            ) {
                let seq = SectionComposition::sequence(vec![
                    SectionComposition::leaf("a", "A", "", p1),
                    SectionComposition::leaf("b", "B", "", p2),
                    SectionComposition::leaf("c", "C", "", p3),
                ]);
                let total = seq.total_estimated_pages();
                let expected = p1 + p2 + p3;
                prop_assert!((total - expected).abs() < 0.01);
            }

            #[test]
            fn prop_constrained_preserves_pages(pages in 0.0f32..50.0, limit in 1u32..100) {
                let c = SectionComposition::constrained(
                    SectionComposition::leaf("x", "X", "", pages),
                    Some(limit),
                    true,
                );
                prop_assert!((c.total_estimated_pages() - pages).abs() < f32::EPSILON);
            }

            #[test]
            fn prop_optional_preserves_pages(pages in 0.0f32..50.0) {
                let opt = SectionComposition::optional(
                    SectionComposition::leaf("x", "X", "", pages),
                );
                prop_assert!((opt.total_estimated_pages() - pages).abs() < f32::EPSILON);
            }

            #[test]
            fn prop_serde_roundtrip(leaf in arb_leaf()) {
                let json = serde_json::to_string(&leaf).unwrap();
                let deser: SectionComposition = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(deser.leaf_count(), leaf.leaf_count());
            }
        }
    }
}
