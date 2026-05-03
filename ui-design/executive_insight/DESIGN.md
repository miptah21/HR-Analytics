---
name: Executive Insight
colors:
  surface: '#f8f9ff'
  surface-dim: '#cbdbf5'
  surface-bright: '#f8f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eff4ff'
  surface-container: '#e5eeff'
  surface-container-high: '#dce9ff'
  surface-container-highest: '#d3e4fe'
  on-surface: '#0b1c30'
  on-surface-variant: '#44474e'
  inverse-surface: '#213145'
  inverse-on-surface: '#eaf1ff'
  outline: '#75777f'
  outline-variant: '#c5c6cf'
  surface-tint: '#4e5e82'
  primary: '#031636'
  on-primary: '#ffffff'
  primary-container: '#1a2b4c'
  on-primary-container: '#8293ba'
  inverse-primary: '#b6c6f0'
  secondary: '#006492'
  on-secondary: '#ffffff'
  secondary-container: '#58bcfd'
  on-secondary-container: '#004a6d'
  tertiary: '#001c0d'
  on-tertiary: '#ffffff'
  tertiary-container: '#00331c'
  on-tertiary-container: '#43a470'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#d8e2ff'
  primary-fixed-dim: '#b6c6f0'
  on-primary-fixed: '#071b3b'
  on-primary-fixed-variant: '#364669'
  secondary-fixed: '#cae6ff'
  secondary-fixed-dim: '#8ccdff'
  on-secondary-fixed: '#001e2f'
  on-secondary-fixed-variant: '#004b6f'
  tertiary-fixed: '#95f7bb'
  tertiary-fixed-dim: '#7adaa1'
  on-tertiary-fixed: '#002110'
  on-tertiary-fixed-variant: '#005230'
  background: '#f8f9ff'
  on-background: '#0b1c30'
  surface-variant: '#d3e4fe'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 36px
    fontWeight: '700'
    lineHeight: 44px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  title-sm:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: '0'
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: '0'
  body-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: '0'
  label-caps:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.05em
  data-mono:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
    letterSpacing: -0.01em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  base: 4px
  xs: 8px
  sm: 16px
  md: 24px
  lg: 32px
  xl: 48px
  container-max: 1440px
  gutter: 24px
  sidebar-width: 260px
---

## Brand & Style
This design system is built on the principles of **Corporate Modernism**, prioritizing clarity, precision, and institutional trust. The aesthetic is designed to feel like a high-performance instrument for HR leadership, stripping away decorative elements in favor of a functional, data-first interface.

The brand personality is intelligent and composed. It avoids the playfulness of consumer SaaS, instead using a structured layout and a constrained color palette to evoke an atmosphere of executive-level decision-making. The UI should feel substantial and reliable, utilizing generous whitespace to prevent information density from becoming overwhelming.

## Colors
The palette is anchored by **Navy Blue (#1A2B4C)**, providing a foundation of authority and stability. **Teal (#2D9CDB)** serves as the primary action and accent color, offering a modern, energetic contrast that draws the eye to key metrics and interactive states.

A curated set of semantic colors (Green for growth/retention, Red for churn/attrition) is utilized strictly for data visualization and status indicators. The background uses a very subtle cool gray to reduce eye strain during long-form data analysis, while primary surfaces remain pure white to maximize contrast and perceived cleanliness.

## Typography
This design system utilizes **Inter** exclusively to ensure a systematic and utilitarian feel across all platforms. The typography is optimized for legibility in complex data tables and dense dashboards.

- **Headlines:** Use tighter letter-spacing and heavier weights to create a strong visual anchor for page sections.
- **Data Mono:** While using Inter, tabular figures should be enabled to ensure numerical alignment in lists and tables.
- **Labels:** Small, uppercase labels with increased tracking are used for metadata and table headers to distinguish them from actionable content.

## Layout & Spacing
The layout follows a **Fixed Grid** philosophy within a max-width container for high-resolution displays, ensuring that dashboards do not become distorted on ultra-wide monitors. A 12-column grid system is used for dashboard layouts, with standard 24px gutters.

The vertical rhythm is based on a 4px baseline grid. Components like cards and data tables should use consistent internal padding (usually 24px) to maintain a sense of openness. Sidebar navigation is fixed to the left, providing a persistent structural frame for the application.

## Elevation & Depth
Depth is communicated through **Tonal Layers** and subtle, low-opacity shadows. This design system avoids heavy shadows to maintain a clean, flat aesthetic that feels modern and efficient.

- **Level 0 (Canvas):** The background layer (#F8FAFC).
- **Level 1 (Cards/Surface):** White surfaces with a 1px border (#E2E8F0) and a very soft 4px blur shadow at 5% opacity.
- **Level 2 (Dropdowns/Modals):** Floating elements use a more pronounced 12px blur shadow at 10% opacity to indicate temporary interaction.
- **Interactions:** Hover states on interactive cards should result in a slight vertical lift (2px) and a subtle increase in shadow density.

## Shapes
The shape language is **Soft (0.25rem)**. This slight rounding takes the edge off the "brutal" corporate aesthetic without appearing overly casual or consumer-focused. 

- **Primary Components:** Buttons, input fields, and small tags use a 4px radius.
- **Containers:** Dashboard cards and modal windows use 8px (rounded-lg) to create a clear visual distinction from the elements contained within them.
- **Data Viz:** Bar charts and progress bars should use subtle rounding (2px) to feel integrated with the overall UI.

## Components
- **Buttons:** Primary buttons use Navy Blue with white text. Secondary buttons use a Teal outline with Teal text. All buttons have a fixed height of 40px for standard actions and 32px for compact table actions.
- **Input Fields:** Use a 1px border (#CBD5E1) that shifts to Teal on focus. Labels are consistently placed above the field in `body-sm` weight.
- **Data Tables:** The core of the design system. Rows utilize a subtle hover state (#F1F5F9). Column headers use `label-caps`. Cell content is strictly aligned: text to the left, numbers to the right.
- **Chips/Status Tags:** Use a "Light Fill" approach—a desaturated background of the semantic color with high-contrast text (e.g., Light Red background with Dark Red text for "High Turnover").
- **KPI Cards:** Feature a large `display-lg` metric, a `title-sm` label, and a small sparkline or trend indicator in the bottom right corner.
- **Sidebar:** Uses a dark theme variant of the primary color (#1A2B4C) to create a strong vertical anchor, with active states highlighted by a Teal vertical bar on the left edge.