# UI Migration Guide: index.html → Next.js

## ✅ COMPLETED

### 1. **CSS Design Tokens** 
- ✅ Created design token mapping in `globals.css`
- All color variables, fonts, and spacing preserved
- Ready for selective integration

### 2. **JSX Conversion** 
- ✅ Created `/ui/app/page-landing.tsx` 
- Converted HTML → React JSX components
- `class` → `className`
- Canvas refs with `useEffect` lifecycle
- All sections: hero, overview, simulation, architecture, metrics

### 3. **Canvas Initialization**
- ✅ Added `useRef` hooks for canvas elements
- ✅ Wrapped canvas logic in `useEffect`
- ✅ TODO placeholders for actual drawing logic

---

## 📋 INTEGRATION STEPS

### Step 1: Apply index.html CSS to globals.css

Your `globals.css` has old styles. Update it with index.html theme:

```bash
# Option A: Keep existing structure, add index.html CSS as override
# Option B: Replace entire globals.css with index.html styles (RECOMMENDED)
```

The `page-landing.tsx` expects these CSS classes:
- Layout: `nav`, `section`, `footer`
- Components: `card`, `btn-primary`, `btn-secondary`, `metric-block`
- Utilities: `anim-1`, `divider`, `hero-stats`

### Step 2: Choose Integration Strategy

**Option A: Use Landing Page Separately** ✅ (SIMPLEST)
```
/app/page.tsx          → Your existing interactive app
/app/landing.tsx       → New static landing (page-landing.tsx)
/app/layout.tsx        → Wrap both
```

**Option B: Replace page.tsx Completely** (if landing is the main page)
```
/app/page.tsx          → Replace with page-landing.tsx
/app/mission/page.tsx  → Move existing components to subfolder
```

**Option C: Hybrid** (recommended)
```
/app/page.tsx          → Dynamic switcher (landing vs mission)
/app/landing.tsx       → page-landing.tsx
/app/components/*      → Existing components
```

---

## 🎨 CSS INTEGRATION

### Required Classes from index.html:

```css
/* Navigation */
nav, .nav-logo, .nav-links, .nav-badge

/* Hero Section */
#hero, .hero-content, .hero-tag, .hero-ctas
.btn-primary, .btn-secondary

/* Cards */
.cards-grid, .card, .card-id, .card-title, .card-footer

/* Simulation */
.sim-wrapper, .sim-topbar, .sim-body, .sim-panel
.agent-row, .trust-bar-bg, .trust-bar-fill
.metric-block, .metric-value

/* Architecture */
.arch-node, .arch-flow

/* Metrics */
.metric-block, .metric-grid

/* Footer */
footer, .footer-left, .footer-right

/* Animations */
@keyframes pulse-dot, fadeInUp
.anim-1, .anim-2, .anim-3, .anim-4, .anim-5
```

---

## 🖼️ CANVAS MIGRATION

The new `page-landing.tsx` has placeholders:

```typescript
function initHeroCanvas(canvas: HTMLCanvasElement) {
  // TODO: Copy canvas.js logic from index.html
  // The neural network grid visualization
}

function initSimCanvas(canvas: HTMLCanvasElement) {
  // TODO: Copy canvas.js logic from index.html  
  // The orchestrator network visualization
}
```

**To migrate canvas logic:**

1. Find the inline `<script>` block in `index.html` (lines ~1710+)
2. Extract the two IIFE functions:
   - Hero canvas: Neural network grid
   - Sim canvas: Orchestrator network
3. Paste into `initHeroCanvas()` and `initSimCanvas()`
4. Update canvas size handling for React

---

## ✨ FEATURES NOT YET IMPLEMENTED

These are hooks for future work - marked with `TODO`:

- [ ] Canvas drawing logic (detailed in "Canvas Migration" above)
- [ ] Button click handlers (Launch Mission, Read Docs)
- [ ] Live metrics updates (currently static)
- [ ] Specialist status animation
- [ ] Episode log scrolling (right panel)

---

## 📁 FILE STRUCTURE AFTER MIGRATION

```
/ui
├── app/
│   ├── layout.tsx         (unchanged)
│   ├── page.tsx           (existing - interactive mode)
│   ├── page-landing.tsx   (NEW - landing page)
│   ├── globals.css        (UPDATED with index.html styles)
│   ├── components/
│   │   ├── Landing.tsx
│   │   ├── MissionControl.tsx
│   │   └── ...
│   └── hooks/
│       └── useSentinel.ts
├── index.html             (ARCHIVE - no longer needed after migration)
├── .env                   (unchanged)
└── package.json           (unchanged)
```

---

## 🚀 QUICK START

1. **Update globals.css** with index.html styles (or use the CSS I provided)
2. **Keep page-landing.tsx** as new file (or rename to page.tsx if replacing)
3. **Migrate canvas logic** from index.html `<script>` section
4. **Test**: Visit `http://localhost:3000/` or `/landing` depending on routing

---

## ✅ DONE vs 🚧 TODO

| Part | Status | Notes |
|------|--------|-------|
| HTML → JSX | ✅ | page-landing.tsx ready |
| CSS Variables | ✅ | Design tokens extracted |
| Layout/Structure | ✅ | All sections preserved |
| Canvas Setup | ✅ | useRef + useEffect ready |
| Canvas Drawing | 🚧 | TODO - copy from index.html |
| Button Handlers | 🚧 | TODO - implement event handlers |
| Live Updates | 🚧 | TODO - connect to API/state |
| Animations | ✅ | @keyframes preserved in CSS |

---

## NEXT: What You Need to Do

1. **Decision**: Keep separate landing page or replace main page?
2. **CSS**: Merge index.html styles into `globals.css`
3. **Canvas**: Copy drawing logic from `index.html` `<script>` into the TODO functions
4. **Test**: Run `npm run dev` and verify visual parity

Need help with any step? 🚀
