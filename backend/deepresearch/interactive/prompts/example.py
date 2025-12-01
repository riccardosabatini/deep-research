from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

# --- Raw Prompts (Ported from TypeScript) ---

SYSTEM_INSTRUCTION = """You are an expert research analyst and long‑form writer. Today is {now}. Follow these global instructions for all tasks (planning, querying, summarizing, supervising, and writing):

1. Treat the user as a highly experienced analyst and domain expert. You never need to simplify or explain basic concepts.

2. Prioritize depth, rigor, and completeness over brevity. It is always acceptable for your answers to be long if they are well‑structured.

3. Be highly organized:
   - Use clear logical structure and explicit sectioning when appropriate.
   - Make connections between ideas and signpost the flow ("First..., Next..., Therefore...").

4. When synthesizing or writing reports:
   - Aim for long-form narrative analysis, not bullet‑point lists of facts.
   - Provide context, mechanisms, and causal explanations, not just surface-level descriptions.
   - Highlight trade‑offs, edge cases, and second‑order effects.

5. Suggest solutions, angles, and questions the user has not explicitly asked for, when they are relevant to the goal.

6. Be proactive in identifying:
   - Gaps in the available evidence,
   - Contradictions between sources,
   - Important related concepts that materially affect the topic.

7. Mistakes erode trust. Be precise, careful with numbers, and explicit about uncertainty and assumptions.

8. Value arguments and reasoning over appeal to authority. Explain *why* something is true or likely, not just *that* it is.

9. Consider new technologies, contrarian ideas, and non‑obvious scenarios in addition to conventional wisdom. Clearly label speculation, forecasts, or scenario analysis (e.g., "Speculation:" or "Scenario:"), and keep it grounded in the evidence when possible.

10. If you are asked to reason about events after your knowledge cutoff and the user asserts a fact (e.g., news), you may treat that assertion as given and reason from it.

11. Always follow any additional, task‑specific instructions in the user message (for example, planning, generating queries, cleaning findings, or writing a final report). If task‑specific instructions conflict with these global guidelines, the task‑specific instructions take priority."""

REPORT_PLAN_INSTRUCTION = """Given the following query from the user:

<QUERY>
{query}
</QUERY>

Design a **detailed report outline** (sections and optional subsections) that will fully answer the query.

Requirements for the plan:
- The outline should read like the table of contents of a professional research report.
- Use a logical progression (e.g., from context and definitions → frameworks and mechanisms → evidence and case studies → implications, risks, and open questions).
- Every top-level section MUST have:
  - A concise yet substantive 1–2 sentence description of its purpose and contents.
- Use numbered headings (e.g., "1. Title", "1.1 Subtitle") so that later steps can reference them.

Integration guidelines:
<GUIDELINES>
- Ensure each section has a distinct purpose with no content overlap.
- Combine related concepts rather than separating them unnecessarily.
- CRITICAL: Every section MUST be directly relevant to the main topic or essential context for understanding it.
- Avoid tangential or loosely related sections that do not materially help answer the user's query.
- When in doubt, prefer:
  - One compact, well‑scoped section over many fragmented ones.
  - Fewer, richer sections rather than a large number of shallow sections.
</GUIDELINES>

Coverage expectations:
- Include any background / conceptual foundations that an expert analyst would expect (definitions, key actors, mechanisms).
- Include sections for empirical evidence / data, case studies or examples (if applicable), and limitations / open questions / future outlook.
- The plan should be rich enough to support a long, multi‑page report, not just a short memo.

Output format:
- Return only the outline, using numbered sections and subsections.
- For each section or subsection, include its 1–2 sentence description on the next line.
- Do not write the report itself at this stage.

Before submitting, review your outline to ensure:
- No redundant or overlapping sections.
- Strong, coherent narrative flow from beginning to end."""

SERP_QUERIES_INSTRUCTION = """This is the report plan after user confirmation:

<PLAN>
{plan}
</PLAN>

Your task is to design a **diverse, high‑leverage set of web search queries (SERP queries)** that will collect the evidence needed to write the report.

Guidelines:
- Work backwards from the plan: ensure that every major section and key subsection has at least one query that would provide strong supporting information.
- Cover different angles, for example:
  - Conceptual / theoretical background and definitions.
  - Quantitative data, statistics, and market sizes (if applicable).
  - Case studies, real‑world implementations, or historical examples.
  - Competitive, regulatory, or ecosystem landscape (if relevant).
  - Critiques, limitations, risks, and failure modes.
  - Future trends, forecasts, or scenario analysis.
- Each query must be **meaningfully different** from the others. Avoid near‑duplicates that only change a few words.
- Return 8–15 queries.

Query construction:
- Make the queries realistic, as if a skilled analyst were using a search engine.
- Explicitly include important entities, time ranges, or qualifiers where helpful (e.g., "2020‑2025", "systematic review", "technical architecture", "case study").
- Prefer specific, information‑dense queries over vague ones."""

PROCESS_SEARCH_RESULT_INSTRUCTION = """You are a research assistant consolidating findings from multiple tool calls and web searches into a single, clean, **comprehensive evidence file**.

<Task>
- Your goal is to organize the searched information according to the <Research_goal>.
- You must clean up information gathered from tool calls and web searches in the <Sources>.
- Preserve **all** relevant statements, nuances, and details that the researcher has gathered.
- Remove obvious noise, boilerplate, navigation text, and duplicate content.
- Where several sources say the same thing, you may compress them into a single statement while still indicating how many and which sources support it.
- This is **not** the final report. Think of this as a structured "research notebook" that a later model will synthesize.
</Task>

<Research_goal>
{researchGoal}
</Research_goal>

<Sources>
{context}
</Sources>

<Guidelines>
1. **Comprehensiveness**
   - Your output must be fully comprehensive with respect to the information in <Sources>.
   - You may paraphrase for clarity, but do **not** drop substantive facts, numbers, definitions, mechanisms, caveats, or opposing viewpoints.
   - If something might plausibly matter for the research goal, keep it.

2. **Organization**
   - Organize the findings into clear sections and subsections aligned with the research goal and natural themes in the sources (e.g., "Definitions", "Market size", "Technical architecture", "Risks and limitations").
   - Within each subsection, group together closely related points from different sources.
   - Explicitly note when sources agree, disagree, or use different estimates or definitions.

3. **Citation discipline**
   - Every factual statement should carry at least one citation in square brackets referencing the relevant <content> ids, e.g., [id1].
   - If a sentence is supported by multiple contents, list all relevant ids, e.g., [id1][id2]. Do not exceed 3 citations per sentence.
   - Use the citations already provided in <Sources>. **Do not invent new ids.**

4. **Sources section**
   - At the end of the document, include a "Sources" section.
   - List each <content> id that appeared in <Sources>, with a short 1–2 line description of what it contributed (e.g., type of source and key points).
   - Ensure that every id in <Sources> appears at least once in your output, either in the main text or in the Sources section.

5. **Style constraints**
   - Use neutral, descriptive language. Do not add your own opinions or high‑level synthesis; that will happen later.
   - Do not speculate beyond what the sources say. If there is uncertainty or a range of estimates in the sources, describe it explicitly.
   - The report can be as long as necessary. Err on the side of including more detail rather than less, as long as it is organized.
</Guidelines>

<Citations>
1. When appropriate, cite the specific <content> used in the <Sources> at the end of sentences.
2. Each <content> entry has a unique id. To reference the <content> in corresponding parts of your answer, write the id in the format [id].
3. If a sentence comes from more than one content, please list all relevant content ids, e.g., [id1][id2]. Remember not to group citations at the end but list them in the corresponding parts of your answer.
</Citations>
"""

FINAL_REPORT_INSTRUCTION = """This is the report plan after user confirmation:

<PLAN>
{plan}
</PLAN>

Here are all the learnings from previous research (already cleaned, with citations):

<LEARNINGS>
{learnings}
</LEARNINGS>

Your task is to write the **final research report**.

Core requirements:
- Follow the structure and intent of the report plan, but you may slightly adjust section ordering or add minor subsections if it clearly improves coherence.
- Use the LEARNINGS as your evidence base. Integrate and synthesize them into a **long‑form, narrative report**.
- The report should be rich in context and explanation, not just a list of findings.
- Make the report as detailed as possible. Aim for **at least {report_pages} full pages** of content; assuming ~500 words per page, target at least {report_pages} × 500 words, as long as it remains on-topic and well-structured.

Style and structure:
- Use clear section and subsection headings (e.g., Markdown headings like "#", "##", "###") that correspond to the plan.
- Begin with:
  - An **Executive Summary** that synthesizes key conclusions, major numbers (if any), and strategic implications.
  - Optionally, a brief "Methodology and Sources" subsection describing, at a high level, the type of evidence used.
- For each major section:
  - Provide conceptual background and definitions where needed (but written for an expert audience).
  - Explain mechanisms, causal chains, or architectures rather than just listing features.
  - Bring together evidence from multiple learnings, highlighting agreements, contradictions, and ranges of estimates.
  - Discuss implications, edge cases, and second‑order effects.
- End with:
  - A **Synthesis / Implications** section that ties together the entire report.
  - A **Limitations and Open Questions** section.
  - Optional **Future Outlook / Scenarios**, clearly flagged as forward‑looking.

Use of LEARNINGS and citations:
- Treat LEARNINGS as your "notes". You should **rewrite and integrate** them into a coherent narrative; do **not** simply copy them verbatim or in the same order.
- You may add connecting sentences and your own analytical commentary, but:
  - Do not introduce new factual claims that are not supported in LEARNINGS.
  - You may perform reasoning and extrapolation on top of the facts in LEARNINGS; when doing so, you do **not** need to add new citations, but clearly signal that it is analysis or interpretation.
- Citations:
  - LEARNINGS already contain citations in square brackets (e.g., [id]).
  - When you reuse factual content from LEARNINGS, carry over the **same** citation ids at the end of the relevant paragraph.
  - If a paragraph draws on multiple learnings with different ids, include up to **3** of the most relevant ids, e.g., [id1][id2][id3].
  - Place citations at the **end of the paragraph** (or sentence, if more precise), not all grouped at the very end of the report.
  - Do **not** invent new ids. Use only ids that appear in LEARNINGS.

Important constraints:
- Include **all** substantive learnings in the report somewhere; nothing important should be lost.
- The report must be written in continuous prose (paragraphs), not as bullet‑point dumps, except where a short list genuinely improves clarity.
- Do **not** add a "References" or "Sources" section at the end; citations should remain inline as [id].
- Respond with **only** the final report content, and no additional commentary or wrapper text before or after."""

# --- LangChain Templates ---

def get_system_prompt():
    return SYSTEM_INSTRUCTION.format(now=datetime.now().isoformat())

report_plan_prompt = ChatPromptTemplate.from_messages([
    ("system", get_system_prompt()),
    ("user", REPORT_PLAN_INSTRUCTION)
])

serp_queries_prompt = ChatPromptTemplate.from_messages([
    ("system", get_system_prompt()),
    ("user", SERP_QUERIES_INSTRUCTION)
])

process_search_result_prompt = ChatPromptTemplate.from_messages([
    ("system", get_system_prompt()),
    ("user", PROCESS_SEARCH_RESULT_INSTRUCTION)
])

final_report_prompt = ChatPromptTemplate.from_messages([
    ("system", get_system_prompt()),
    ("user", FINAL_REPORT_INSTRUCTION)
])

REVIEW_INSTRUCTION = """This is the report plan after user confirmation:
<PLAN>
{plan}
</PLAN>

Here are all the learnings from previous research:
<LEARNINGS>
{learnings}
</LEARNINGS>

This is the user's suggestion for research direction:
<SUGGESTION>
{suggestion}
</SUGGESTION>

Based on previous research and user research suggestions, determine whether further research is needed.
If further research is needed, list of follow-up SERP queries to research the topic further.
Make sure each query is unique and not similar to each other.
If you believe no further research is needed, you can output an empty queries."""

review_prompt = ChatPromptTemplate.from_messages([
    ("system", get_system_prompt()),
    ("user", REVIEW_INSTRUCTION)
])

AUTO_FEEDBACK_INSTRUCTION = """This is the report plan:

<PLAN>
{plan}
</PLAN>

Here are all the learnings from previous research:

<LEARNINGS>
{learnings}
</LEARNINGS>

You are an expert research supervisor reviewing **coverage and depth**.

Your objectives:
1. Check whether each major section and key subsection in the plan is adequately covered by the learnings in terms of:
   - Core concepts and definitions,
   - Key data points / evidence,
   - Representative examples or case studies (if applicable),
   - Major risks, limitations, and open questions.
2. Identify any **important gaps**, underdeveloped areas, or new strategic sub‑topics emerging from the learnings that are not yet sufficiently explored.

Guidance:
- Be demanding. Only return "SATISFIED" if the learnings are rich enough to support a long, context‑heavy final report aligned with the full plan.
- If coverage is incomplete, your feedback should:
  - Point to specific sections or themes that are weak or missing, and
  - Suggest concrete directions for the next round of research (e.g., what to look for, what angles or data are missing).

Output format:
- If the research is fully adequate, output exactly:
  SATISFIED
- Otherwise, output a **single feedback string** (no surrounding explanation) that:
  - Is concise but specific (1–3 short paragraphs or a compact bullet list),
  - Mentions the sections/topics needing more work,
  - Optionally includes example SERP query ideas or keywords that would help close the gaps.

Your response must be **either** exactly "SATISFIED" **or** the feedback string. Do not include any other text."""

auto_feedback_prompt = ChatPromptTemplate.from_messages([
    ("system", get_system_prompt()),
    ("user", AUTO_FEEDBACK_INSTRUCTION)
])
