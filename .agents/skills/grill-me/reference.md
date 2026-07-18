# Grill Me — Reference

Detailed question scripts, pushback patterns, and design doc templates. Read when running Phase 2A, 2B, or Phase 5.

---

## Startup Mode: Six Forcing Questions

### Pushback patterns

**Vague market → force specificity**
- BAD: "That's a big market! Let's explore what kind of tool."
- GOOD: "There are 10,000 AI developer tools right now. What specific task does a specific developer waste 2+ hours on per week that your tool eliminates? Name the person."

**Social proof → demand test**
- BAD: "That's encouraging! Who specifically have you talked to?"
- GOOD: "Loving an idea is free. Has anyone offered to pay? Has anyone asked when it ships? Has anyone gotten angry when your prototype broke? Love is not demand."

**Platform vision → wedge challenge**
- BAD: "What would a stripped-down version look like?"
- GOOD: "That's a red flag. If no one gets value from a smaller version, the value prop isn't clear yet. What's the one thing a user would pay for this week?"

**Growth stats → vision test**
- BAD: "That's a strong tailwind."
- GOOD: "Growth rate is not a vision. Every competitor cites the same stat. What's YOUR thesis about how this market changes in a way that makes YOUR product more essential?"

**Undefined terms → precision demand**
- BAD: "What does your current onboarding flow look like?"
- GOOD: "'Seamless' is not a product feature. What specific step causes drop-off? What's the rate? Have you watched someone go through it?"

---

### Q1: Demand Reality

**Ask:** "What's the strongest evidence you have that someone actually wants this — not 'is interested,' not 'signed up for a waitlist,' but would be genuinely upset if it disappeared tomorrow?"

**Push until you hear:** Specific behavior. Someone paying. Someone expanding usage. Someone building their workflow around it.

**Red flags:** "People say it's interesting." "We got 500 waitlist signups." "VCs are excited." None of these are demand.

**After first answer, check framing:**
1. **Language precision:** Challenge undefined terms — "What do you mean by [term]? Can you define it so I could measure it?"
2. **Hidden assumptions:** Name one assumption and ask if it's verified.
3. **Real vs hypothetical:** "I think developers would want..." is hypothetical. "Three developers spent 10 hours a week on this" is real.

If imprecise, reframe: "Let me try restating what I think you're actually building: [reframe]. Does that capture it better?"

---

### Q2: Status Quo

**Ask:** "What are your users doing right now to solve this problem — even badly? What does that workaround cost them?"

**Push until you hear:** A specific workflow. Hours spent. Dollars wasted. Tools duct-taped together. People hired to do it manually.

**Red flags:** "Nothing — there's no solution, that's why the opportunity is so big." If truly nothing exists, the problem probably isn't painful enough.

---

### Q3: Desperate Specificity

**Ask:** "Name the actual human who needs this most. What's their title? What gets them promoted? What gets them fired? What keeps them up at night?"

**Push until you hear:** A name. A role. A specific consequence. Ideally something heard directly from that person.

**Red flags:** "Healthcare enterprises." "SMBs." "Marketing teams." Categories, not people.

**Forcing exemplar:**

SOFTENED (avoid): "Who's your target user, and what gets them to buy?"

FORCING (aim for): "Name the actual human. Not 'product managers at mid-market SaaS' — an actual name, title, consequence. What's the real thing they're avoiding? If you can't name them, you don't know who you're building for — and 'users' isn't an answer."

Match consequence to domain: B2B → career impact; consumer → daily pain; hobby → weekend project unlocked.

---

### Q4: Narrowest Wedge

**Ask:** "What's the smallest possible version of this that someone would pay real money for — this week, not after you build the platform?"

**Push until you hear:** One feature. One workflow. Something shippable in days that someone would pay for.

**Red flags:** "We need the full platform first." "Stripped down wouldn't be differentiated."

**Bonus push:** "What if the user didn't have to do anything to get value? No login, no integration, no setup. What would that look like?"

---

### Q5: Observation & Surprise

**Ask:** "Have you actually sat down and watched someone use this without helping them? What did they do that surprised you?"

**Push until you hear:** A specific surprise contradicting the founder's assumptions.

**Red flags:** "We sent a survey." "We did demo calls." "Nothing surprising, going as expected." Surveys lie. Demos are theater.

**The gold:** Users doing something the product wasn't designed for — often the real product emerging.

---

### Q6: Future-Fit

**Ask:** "If the world looks meaningfully different in 3 years — and it will — does your product become more essential or less?"

**Push until you hear:** A specific claim about how users' world changes and why that makes the product more valuable.

**Red flags:** "The market is growing 20% per year." "AI will make everything better." Rising tide arguments every competitor can make.

---

**Smart-skip:** If earlier answers already cover a later question, skip it.

---

## Builder Mode Questions

Ask ONE AT A TIME. Generative, not interrogative.

- **What's the coolest version of this?** What would make it genuinely delightful?
- **Who would you show this to?** What would make them say "whoa"?
- **What's the fastest path to something you can actually use or share?**
- **What existing thing is closest to this, and how is yours different?**
- **What would you add if you had unlimited time?** What's the 10x version?

**Wild exemplar:**

STRUCTURED (avoid): "Consider adding a share feature for retention."

WILD (aim for): "What if you let them share the visualization as a live URL? Or pipe it into Slack? Or animate the generation so viewers see it draw itself? Each one's a 30-minute unlock."

---

## Design Doc Templates

### Startup mode

```markdown
# Design: {title}

Generated by /grill-me on {date}
Branch: {branch}
Status: DRAFT
Mode: Startup
Supersedes: {prior filename — omit if first on branch}

## Problem Statement
{from Phase 2A}

## Demand Evidence
{from Q1}

## Status Quo
{from Q2}

## Target User & Narrowest Wedge
{from Q3 + Q4}

## Constraints
{from Phase 2A}

## Premises
{from Phase 3}

## Approaches Considered
### Approach A: {name}
{from Phase 4}
### Approach B: {name}
{from Phase 4}

## Recommended Approach
{chosen approach with rationale}

## Open Questions
{unresolved}

## Success Criteria
{measurable from Phase 2A}

## Distribution Plan
{how users get the deliverable; CI/CD or deferral}
{omit if web service with existing deploy pipeline}

## Dependencies
{blockers, prerequisites}

## The Assignment
{one concrete real-world action — not "go build it"}

## What I noticed about how you think
{2-4 bullets quoting specific things the user said}
```

### Builder mode

```markdown
# Design: {title}

Generated by /grill-me on {date}
Branch: {branch}
Status: DRAFT
Mode: Builder
Supersedes: {prior filename — omit if first on branch}

## Problem Statement
{from Phase 2B}

## What Makes This Cool
{core delight / "whoa" factor}

## Constraints
{from Phase 2B}

## Premises
{from Phase 3}

## Approaches Considered
### Approach A: {name}
### Approach B: {name}

## Recommended Approach
{chosen approach with rationale}

## Open Questions
{unresolved}

## Success Criteria
{what "done" looks like}

## Distribution Plan
{how users get it}

## Next Steps
{concrete build tasks — first, second, third}

## What I noticed about how you think
{2-4 bullets quoting specific things the user said}
```
