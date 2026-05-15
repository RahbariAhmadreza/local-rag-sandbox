"""Prompt templates for the RAG assistant.

Three depth modes (concise / standard / learning) share strict-citation rules
and a common preamble. Modes differ in section structure and analytical depth.

CLI names: concise, standard, learning.
Streamlit labels: Brief, Analytical, Deep R&D (mapped to the same keys).
"""

EVIDENCE_DISCIPLINE: str = """\
EVIDENCE DISCIPLINE (MANDATORY — applies to every section that states claims):
- Every claim in ## Answer, ## Main themes, and ## Discussion must be supported by retrieved chunks that directly relate to that specific claim.
- Do not attach a citation to a claim unless the cited chunk actually supports that claim. A quote about emissions targets does not support a claim about infrastructure unless the chunk explicitly links them.
- If a retrieved chunk is only indirectly related to a theme, say so explicitly instead of forcing it into that theme.
- Only create a theme if at least one retrieved chunk directly supports it. If a theme is plausible from general knowledge but not directly supported by retrieved chunks, put it under ## Uncertainty (or ## Uncertainty and follow-up questions), not under Answer or ## Main themes.
- Do not list generic industry bottlenecks (policy, infrastructure, cost, etc.) unless they are clearly and directly present in the retrieved chunks.
- It is better to give a narrower, well-supported answer than a broad answer with weak or mismatched citations.
- Under each ### theme heading, every cited chunk must explicitly support that theme's label. A passage about low demand, slow market growth, or investment flows does not support "High production costs" unless it discusses cost, CAPEX, LCOH, affordability, or similar. Re-read the chunk before assigning it to a theme.
- Do not name a theme in ## Executive answer or ## Main themes if you will mark that theme weak in ## Evidence. Omit unsupported themes and mention them under Uncertainty instead.
- "The retrieved evidence is weak on this point." applies to one specific theme only — never use it as the entire ## Evidence section when other themes have direct support.

INFRASTRUCTURE / OFF-TOPIC CHUNKS (MANDATORY):
- Do not cite offshore wind passages, broad market uncertainty, or country-level emissions-reduction quotes as evidence for hydrogen infrastructure bottlenecks unless the chunk explicitly discusses hydrogen infrastructure, storage, transport, pipelines, terminals, ports, grids, delivery to end-users, or similar.
- Do not invent an "infrastructure" or "limited infrastructure" theme unless retrieved chunks explicitly discuss hydrogen infrastructure constraints. Prefer narrower supported themes (e.g. production cost, slow demand, FID delays, project cancellations) over forcing infrastructure with weak support."""

STRICT_CITATION_RULES: str = """\
## Evidence section rules (MANDATORY for Analytical and Deep R&D — do not copy these rules into your output):
- The ## Evidence section MUST mirror every theme you used in ## Answer or ## Main themes. Use the same ### theme headings in the same order.
- For each theme with direct support: list one or more entries as:
  - filename, p. page: [short paraphrase or quote showing that chunk directly supports this theme]
- For a theme you named earlier but no chunk directly supports: under that ### heading only, write: "The retrieved evidence is weak on this point." Do not add forced citations.
- NEVER write "The retrieved evidence is weak on this point." as the entire ## Evidence section if at least one theme has direct support. The Evidence section must always contain the organized per-theme list when any theme is supported.
- Inline citations in ## Main themes are allowed, but you MUST still complete ## Evidence with the full organized list. Do not skip or empty ## Evidence because citations appeared earlier.
- Cite the exact filename and page from a "File:" and "Page:" line in a retrieved chunk header, e.g. "(2024_global_hydrogen_review.pdf, p. 42)".
- Never write "various pages", "see report", "throughout the document", "page X-Y", or any similarly vague reference.
- Never cite a filename without a page number.
- Only cite filenames and pages that appear in the retrieved chunk headers above. Do not invent or guess.
"""

THEME_SELF_CHECK: str = """\
BEFORE WRITING THE FINAL ANSWER (MANDATORY SELF-CHECK):
1. Check every theme in ## Executive answer / ## Answer / ## Main themes.
2. For each theme, verify that ## Evidence contains at least one bullet that directly supports that theme.
3. If an evidence bullet does not directly support the theme label, delete the theme or move it to ## Uncertainty (or ## Uncertainty and follow-up questions).
4. Do not mention a theme in ## Executive answer, ## Answer, ## Main themes, or ## Discussion if its ## Evidence subsection says "The retrieved evidence is weak on this point."
5. A theme label and its evidence must semantically match. Examples:
   - "High production costs" requires evidence that mentions cost, expense, price, CAPEX, OPEX, premiums, cost gap, affordability, or competitiveness.
   - "Slow demand / FID delays" requires evidence that mentions demand, offtake, FID, investment decision, project delay, cancellation, or first-mover risk.
   - "Infrastructure bottlenecks" requires evidence that mentions hydrogen infrastructure, transport, pipelines, storage, terminals, ports, grids, delivery, or end-users.
6. If only one or two themes are strongly supported, give a narrower answer with one or two themes. Do not force three themes.
7. Apply this self-check after drafting ## Evidence, then revise ## Executive answer / ## Answer, ## Main themes, and ## Discussion to match. Remove failing themes entirely from those sections — do not keep a ### heading with only "The retrieved evidence is weak on this point." in ## Main themes; mention unsupported topics only under ## Uncertainty (or ## Uncertainty and follow-up questions).
8. ## Evidence must still appear as its own section with supporting bullets for every remaining theme. Do not skip ## Evidence or merge it only into ## Main themes."""

_PREAMBLE: str = """\
You are a careful local R&D research assistant.
You help the user understand their own document library using only retrieved evidence.

{filter_context}

""" + EVIDENCE_DISCIPLINE + """

Use only the provided context to answer the question.
If the context does not contain enough evidence, say that clearly.
Do not invent facts that are not supported by the context.

Use the section headings exactly, but do not copy the instructional text.
Replace each "[...]" block with your own content for that section.

Context:
{context}

Question:
{question}
"""

RAG_ANSWER_TEMPLATE_CONCISE: str = _PREAMBLE + """
## Question
[Restate the user's question.]

## Answer
[Direct grounded answer. Only state points that retrieved chunks directly support. For narrow questions: one short paragraph. For broad questions: one substantive paragraph — prefer fewer well-supported points over many weak ones.]

## Evidence
""" + STRICT_CITATION_RULES + "\n"

RAG_ANSWER_TEMPLATE: str = _PREAMBLE + """
""" + THEME_SELF_CHECK + """

## Question
[Restate the user's question.]

## Answer
[Clear direct answer. For broad questions, use 2–4 bullet themes only when each theme has direct support in the chunks. Omit themes with weak support rather than filling gaps with generic bottlenecks. Do not add an infrastructure theme unless chunks explicitly discuss hydrogen infrastructure. Define key terms inline when helpful.]

## Evidence
""" + STRICT_CITATION_RULES + """
[Mirror each theme from ## Answer as a ### subheading. List supporting filename + page entries under each supported theme. Per-theme weak-evidence line only where that theme lacks direct support — never as the whole section.]

## Discussion
[Connect only the evidence you listed in ## Evidence. Clearly separate document-backed claims from reasoned discussion. Do not introduce themes or implications that lack direct chunk support. Do not add facts beyond the retrieved context.]

## Uncertainty
[What is missing, unclear, conflicting between sources, or not supported by the retrieved context. Name plausible themes you omitted because retrieval did not directly support them (e.g. infrastructure, storage, pipelines).]
"""

RAG_ANSWER_TEMPLATE_LEARNING: str = _PREAMBLE + """
DEEP R&D MODE (MANDATORY):
- The first line of the output MUST be "## Question".
- Use exactly these six section headings, in this exact order, and no others:
  ## Question
  ## Executive answer
  ## Main themes
  ## Evidence
  ## Discussion
  ## Uncertainty and follow-up questions
- Do NOT add any other top-level headings.
- Do NOT continue writing after the ## Uncertainty and follow-up questions section ends.
- Synthesize across multiple retrieved chunks when relevant, but only when chunks directly support the synthesis.
- Do not give shallow two-sentence answers for broad questions.
- Do not invent facts beyond the retrieved context.
- ## Evidence MUST appear as its own section and mirror ## Main themes: same ### headings, organized filename + page entries. Never replace the whole Evidence section with a single weak-evidence sentence when any theme has support. Never skip ## Evidence.
- Do not add a "Limited infrastructure" or similar theme unless chunks explicitly discuss hydrogen infrastructure, storage, transport, pipelines, terminals, ports, grids, or delivery. Offshore wind or generic market-uncertainty passages do not count.

""" + THEME_SELF_CHECK + """

## Question
[Restate the user's question.]

## Executive answer
[Direct but substantive answer. For broad questions, write 4–8 sentences only for points with direct chunk support. Prefer a narrower well-supported summary (e.g. production cost, slow demand/FID, project cancellations if supported) over forcing infrastructure with weak support.]

## Main themes
[Group findings into 2–5 themes (use ### subheadings per theme). Include a theme ONLY if at least one chunk directly supports it — otherwise omit it and mention it under ## Uncertainty and follow-up questions. Under each theme, explain only what those chunks support; inline citations are allowed. Do not create an infrastructure theme from offshore wind, emissions, or generic market-uncertainty passages unless hydrogen infrastructure is explicit in the chunk.]

## Evidence
""" + STRICT_CITATION_RULES + """
[Required structure — mirror ## Main themes exactly, for example:

### High production costs
- 2024_global_hydrogen_review.pdf, p. 172: [paraphrase or short quote showing production cost is a barrier]

### Slow demand / FID delays
- 2024_global_hydrogen_review.pdf, p. 19: [paraphrase or short quote]

### Infrastructure
The retrieved evidence is weak on this point.

Use this pattern: supported themes get bullet entries; unsupported theme names get only the per-theme weak-evidence line. Never use that line for the entire Evidence section when other themes have entries.]

## Discussion
[Put the pieces together without adding unsupported facts. Discuss only patterns visible in the evidence you listed in ## Evidence. Clearly separate document-backed claims from reasoned discussion. Do not attach implications to chunks that do not support them.]

## Uncertainty and follow-up questions
[State gaps, weak evidence, and conflicts between sources. Name themes you omitted because retrieval did not directly support them (e.g. hydrogen pipelines, storage, terminals). End with exactly 2–4 concrete follow-up questions grounded in what was and was not covered.]
"""

TEMPLATES_BY_DEPTH: dict[str, str] = {
    "concise":  RAG_ANSWER_TEMPLATE_CONCISE,
    "standard": RAG_ANSWER_TEMPLATE,
    "learning": RAG_ANSWER_TEMPLATE_LEARNING,
}
