# Model Training Strategy

The goal is not to make Gemma memorize every rare aircraft. That would go stale quickly and miss new special liveries, retirements, unusual visits, and local context.

The goal is to train Gemma to apply a spotter-oriented rarity judgment:

```text
Would an aviation enthusiast in this area consider this aircraft uncommon, noteworthy, or chase-worthy?
```

## What The Model Should Learn

Train the model on rarity factors, not just aircraft IDs:

- Low production count or few remaining active examples.
- Near end-of-life or disappearing commercial/military type.
- Limited operator, government, VIP, military, test, research, or special mission use.
- Unique shape, role, livery, or individual airframe.
- Geographic infrequency for the user's area.
- Historical or enthusiast value.
- Ordinary local traffic should stay not rare unless another factor overrides it.

## What Still Needs Structured Context

The model cannot infer facts that are absent from the live aircraft state. If the ADS-B feed only says `B738`, it cannot know whether the aircraft has a special livery unless we provide registration or a compact context entry.

Keep a small, updateable context bundle:

- Notable type examples.
- Known special registrations.
- Local airport baseline traffic.
- User watchlists.
- Recently discovered false positives and false negatives.

This is not a full database lookup for every aircraft. It is a compact reference pack that travels with the prompt or is compiled into the fine-tuning examples.

## Dataset Shape

Use several kinds of examples:

- Positive obvious: Beluga, Dreamlifter, An-124, MD-11, DC-8, C-17, H-60, E-4B-like aircraft.
- Positive contextual: ordinary-looking type with special registration or unusual operator.
- Positive local anomaly: heavy or rare visitor at a place where it is not common.
- Negative ordinary: A320/737-family airline traffic at LAX/SAN/SNA/BUR/ONT.
- Negative near miss: Boeing/Airbus manufacturer names alone are not enough.
- Hard negative: common military trainer or common helicopter if it is locally routine and not otherwise notable.

## Training Loop

1. Collect live SoCal snapshots.
2. Weak-label with explicit heuristics.
3. Sample likely false negatives for review, especially uncommon types and non-airline operators.
4. Add manual corrections to reviewed data.
5. Fine-tune on compact JSON prompts with structured JSON outputs.
6. Evaluate on held-out snapshots and known chase-worthy examples.
7. Log production disagreements and feed them back into the reviewed set.

## Prompt Contract

Input should include:

- Normalized aircraft state.
- User/location context.
- Rarity factors.
- A compact list of examples, not an exhaustive master list.

Output should always be:

```json
{
  "is_rare": true,
  "confidence": 0.91,
  "reason": "Short explanation grounded in the aircraft fields and rarity factors."
}
```

## Success Criteria

The model is useful when it:

- Rejects ordinary hub traffic even when the aircraft is large or from Boeing/Airbus.
- Flags chase-worthy classics, military/special mission aircraft, and limited-run aircraft.
- Uses registration/special-livery context when provided.
- Explains uncertainty instead of inventing missing facts.
- Improves from reviewer corrections without needing a complete aircraft encyclopedia.

