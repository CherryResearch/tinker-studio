# Interview Assistant Guide

Use this guide when an assistant is interviewing a person so their answers can become local Tinker Studio training material. The goal is not a polished article. The goal is natural, consented, high-signal chunks that preserve the person's voice and can be written into `raw/interview_qa_raw.jsonl` and `processed/interview_qa.jsonl`.

## Ground Rules

- Ask one question at a time.
- Keep raw answers intact unless the person asks for cleanup.
- Lightly edit only for typos, privacy redaction, and obvious transcription errors.
- Do not invent biographical details, opinions, memories, or examples.
- If the answer includes private names, locations, medical details, passwords, tokens, or non-consenting third parties, ask whether to omit, generalize, or redact it before saving.
- Prefer 80-350 word answers. Shorter is fine for aphorisms, poetry, or reply-style examples.
- Preserve line breaks for poetry, fragments, lists, and deliberately shaped prose.

## Session Flow

1. Explain the purpose: "I am collecting a few answers in your voice so this local app can chunk them into training examples."
2. Ask whether the person wants shortform posts, long-form reflective answers, reply-style material, poetry, or a mix.
3. Collect 3-8 answers in one theme before switching topics.
4. For each answer, ask one follow-up if the first response is abstract, generic, or too short.
5. Before saving, summarize the question, theme, tags, and edited answer. Ask for corrections.
6. Save with `collect_interview_qa.py`, then run review.

## Template Questions

Use these as starting points. Adapt the wording to the person.

- What is a thing people consistently misunderstand about you, your work, or your taste?
- What is a belief you hold that sounds obvious to you but seems strange to other people?
- What kind of writing or conversation makes you immediately trust someone?
- What do you notice first when you are judging whether an idea is real, fake, useful, or dangerous?
- Describe a recent moment where your opinion changed. What caused the update?
- What is something you keep trying to explain, but never quite get across?
- What do you find beautiful, funny, annoying, or sacred that other people underrate?
- Give me a reply you might write to someone who is almost right but missing the point.
- If this were a short post, what would the sharpest version be?
- If this were a longer note, what context would you add?
- What should an assistant never imitate about you?
- What should an assistant preserve even when it is inconvenient or weird?

## Conditional Directions

- If the answer is abstract, ask: "Can you give me one concrete example?"
- If the answer is generic, ask: "What is the version of that only you would say?"
- If the answer is too polished, ask: "Can you say it more casually, like you would in a message?"
- If the answer is too short, ask: "Can you expand that by 3-5 sentences without making it formal?"
- If the answer is too long, ask: "What are the 2-3 chunks that should be preserved?"
- If the person gives poetry, preserve line breaks and ask for a theme label instead of forcing an explanation.
- If the person gives a thread or argument, ask for the thesis, the emotional texture, and one memorable example.
- If the person gives reply-style material, set `--is-reply-style` when saving.
- If the person hesitates or flags privacy risk, skip the item or save a generalized version only after confirmation.

## Save And Review

List built-in prompt rounds:

```powershell
.\tinker_env\Scripts\python.exe .\collect_interview_qa.py --workspace . list-rounds
```

Append one answer. Omitting fields starts interactive prompts:

```powershell
.\tinker_env\Scripts\python.exe .\collect_interview_qa.py --workspace . append --theme "taste and judgment" --tags taste judgment
```

Append one answer with a follow-up:

```powershell
.\tinker_env\Scripts\python.exe .\collect_interview_qa.py --workspace . followup --theme "reply style" --is-reply-style --tags replies shortform
```

Review recent processed rows:

```powershell
.\tinker_env\Scripts\python.exe .\collect_interview_qa.py --workspace . review --last 5
```

Export derived openings grouped by theme:

```powershell
.\tinker_env\Scripts\python.exe .\collect_interview_qa.py --workspace . export-prompts
```

The collector writes into the ignored dataset folder. The run manager later turns each processed row into direct Q&A examples plus compact post-style continuation chunks.
