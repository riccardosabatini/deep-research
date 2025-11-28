from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

# --- Raw Prompts (Ported from TypeScript) ---

SYSTEM_INSTRUCTION = """You are an expert researcher. Today is {now}. Follow these instructions when responding:

- You may be asked to research subjects that is after your knowledge cutoff, assume the user is right when presented with news.
- The user is a highly experienced analyst, no need to simplify it, be as detailed as possible and make sure your response is correct.
- Be highly organized.
- Suggest solutions that I didn't think about.
- Be proactive and anticipate my needs.
- Treat me as an expert in all subject matter.
- Mistakes erode my trust, so be accurate and thorough.
- Provide detailed explanations, I'm comfortable with lots of detail.
- Value good arguments over authorities, the source is irrelevant.
- Consider new technologies and contrarian ideas, not just the conventional wisdom.
- You may use high levels of speculation or prediction, just flag it for me."""

REPORT_PLAN_INSTRUCTION = """Given the following query from the user:
<QUERY>
{query}
</QUERY>

Generate a list of sections for the report based on the topic and feedback.
Your plan should be tight and focused with NO overlapping sections or unnecessary filler. Each section needs a sentence summarizing its content.

Integration guidelines:
<GUIDELINES>
- Ensure each section has a distinct purpose with no content overlap.
- Combine related concepts rather than separating them.
- CRITICAL: Every section MUST be directly relevant to the main topic.
- Avoid tangential or loosely related sections that don't directly address the core topic.
</GUIDELINES>

Before submitting, review your structure to ensure it has no redundant sections and follows a logical flow."""

SERP_QUERIES_INSTRUCTION = """This is the report plan after user confirmation:
<PLAN>
{plan}
</PLAN>

Based on previous report plan, generate a list of SERP queries to further research the topic. Make sure each query is unique and not similar to each other."""

PROCESS_SEARCH_RESULT_INSTRUCTION = """Given the following contexts from a SERP search for the query:
<QUERY>
{query}
</QUERY>

You need to organize the searched information according to the following requirements:
<RESEARCH_GOAL>
{researchGoal}
</RESEARCH_GOAL>

The following context from the SERP search:
<CONTEXT>
{context}
</CONTEXT>

You need to think like a human researcher.
Generate a list of learnings from the contexts.
Make sure each learning is unique and not similar to each other.
The learnings should be to the point, as detailed and information dense as possible.
Make sure to include any entities like people, places, companies, products, things, etc in the learnings, as well as any specific entities, metrics, numbers, and dates when available. The learnings will be used to research the topic further.

Citation Rules:
- Please cite the context at the end of sentences when appropriate.
- Please use the format of citation number [number] to reference the context in corresponding parts of your answer.
- If a sentence comes from multiple contexts, please list all relevant citation numbers, e.g., [1][2]. Remember not to group citations at the end but list them in the corresponding parts of your answer."""

FINAL_REPORT_INSTRUCTION = """This is the report plan after user confirmation:
<PLAN>
{plan}
</PLAN>

Here are all the learnings from previous research:
<LEARNINGS>
{learnings}
</LEARNINGS>

Here are all the sources from previous research, if any:
<SOURCES>
{sources}
</SOURCES>

Here are all the images from previous research, if any:
<IMAGES>
{images}
</IMAGES>

Write a final report based on the report plan using the learnings from research.
Make it as detailed as possible, aim for 5 pages or more, the more the better, include ALL the learnings from research.
**Respond only the final report content, and no additional text before or after.**

Image Rules:
- Images related to the paragraph content at the appropriate location in the article according to the image description.
- Include images using `![Image Description](image_url)` in a separate section.
- **Do not add any images at the end of the article.**

Citation Rules:
- Please cite research references at the end of your paragraphs when appropriate.
- If the citation is from the reference, please **ignore**. Include only references from sources.
- Please use the reference format [number], to reference the learnings link in corresponding parts of your answer.
- If a paragraphs comes from multiple learnings reference link, please list all relevant citation numbers, e.g., [1][2]. Remember not to group citations at the end but list them in the corresponding parts of your answer. Control the number of footnotes.
- Do not have more than 3 reference link in a paragraph, and keep only the most relevant ones.
- **Do not add references at the end of the report.**"""

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
