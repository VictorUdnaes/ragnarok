from langchain_core.prompts import PromptTemplate

# PLANNING ---------------------------------------------------------------------------------------------------------

anonymizer_prompt = """
You anonymize questions in Norwegian by replacing named entities with variables like X, Y, Z, etc.
Examples:
- "Who is the leader in the political party Venstre?" -> "Who is the leader in the political party X?", {{"X": "Venstre"}}
- "Did the leader of the political party Fremskrittspartiet vote in favour of the police reform bill?" -> Did the leader of the political party X vote in favour of the Y bill?". {{"X": "Fremskrittspartiet", "Y": "police reform}}

Input: {question}

Output:
You will put the anonymized question in the "anonymized_question" field, the mapping of variables to original entities in the "mapping" field, and an explanation of your choices in the "explaination" field.
The anonymized_question field should not contain any named entities, only variables like X, Y, Z, etc.
"""

planner_prompt = """
Instructions:
You are an expert in planning information retrieval tasks. Your goal is to create a structured step-by-step plan to answer the given question by breaking it down into actionable steps.

Question: {question}

You exist as part of a RAG chain, so the question will be anonymized, and you will not have access to the original entities. Variables like X, Y, Z are concrete values that are known to the rest of the execution chain 
and will be replaced with the actual entity name (e.g., "France"). Plan as if you already know what Z represents.

CRITICAL: Do NOT include any steps about identifying, resolving, or determining what variables represent. The variables will already be resolved when your plan executes.

Output:
A step-by-step plan to answer this question. 

Requirements:
- Each step should be a distinct, actionable information retrieval or processing task
- Steps should build logically toward the final answer
- The final step should produce the complete answer
- Keep steps concise but complete (typically 2-5 steps total)
"""

deanonymize_prompt = """
You are a text replacement specialist. Your job is to replace variable placeholders with actual values in a structured plan.

Original Plan:
{plan}

Variable Mappings:
{mapping}

Instructions:
1. Replace ALL occurrences of variables (X, Y, Z, etc.) with their corresponding actual values
2. Only replace the variables, do not modify any other part of the structure or content
3. Maintain the exact same JSON structure and all other fields unchanged
4. Variables should be replaced wherever they appear in any text field

Return the updated plan with variables replaced:
"""

# QUERY AUGMENTATION -----------------------------------------------------------------------------------------------

multi_query_gen_prompt = """
You are an expert in Norwegian political analysis and document search. Your task is to generate five strategic query variants of the given question to retrieve the most relevant documents from a vector database containing:

- Political platforms and party manifestos  
- Parliamentary votes and voting records  
- Political statements and press releases  
- Party leader debates and speeches  
- Case documents and legislative proposals  

STRATEGIC APPROACH:  
Generate five different query variants that cover various informational aspects:

1. PROGRAMMATIC QUERY VARIANT - Focus on the party's official positions and platforms  
2. VOTING HISTORY QUERY VARIANT - Focus on actual votes and voting behavior  
3. STATEMENT QUERY VARIANT - Focus on public statements and communication  
4. POLICY AREA QUERY VARIANT - Focus on the specific political issue/topic  
5. COMPARATIVE QUERY VARIANT - Focus on comparison and consistency over time  

GUIDELINES FOR EACH QUERY VARIANT:  
- Use relevant Norwegian political terms and concepts  
- Include both formal and informal expressions  
- Vary wording and perspective significantly  
- Ensure each variant can retrieve unique but relevant documents  
- Focus on information that can reveal consistency/inconsistency  

EXAMPLES OF VARIATION TECHNIQUES:  
- Replace party names with ideological labels (social democrats, liberals, conservatives)  
- Use both formal policy names and everyday descriptions  
- Include both historical and current context  
- Vary between specific and general phrasings  

Original question: {question}

Generate five strategic query variants (one per line):
"""

query_optimization_prompt = """
You are an expert at rewriting user queries to optimize them for semantic similarity search in vector databases.
Transform the user's query into a single optimized version that:

- Uses semantically rich terminology while maintaining the original intent
- Includes relevant keywords and synonyms that might appear in documents
- Converts questions to declarative statements when appropriate
- Adds implicit context that users might assume
- Uses terminology that commonly appears in documentation

Rewrite this query: {query}
"""

queries_from_plan_prompt = """
Instructions:
You are an expert at translating steps of an information retrieval plan into precise vector database queries in Norwegian. Your goal is to generate optimal search queries that will retrieve the information needed to complete each step of the given plan.

Original Question: {question}
Plan: {plan}

For each step in the plan, generate 1-3 specific queries that would retrieve the relevant information from a vector database. 

Guidelines:
- Queries should be specific and focused on retrieving factual information
- Use natural language that would match how information is typically written/stored
- Consider synonyms and alternative phrasings for key concepts
- Each query should target a specific piece of information needed for that step
- Prioritize precision over broad coverage - better to have focused queries than overly general ones

Example:
Original Question: Does the statement from politician X match with current party Y line?
Plan:
1. Determine what party politician Jonas Gahr Støre belongs to
2. Identify the current official positions/policies of party AP
3. Compare the politician's statement with party positions

Step 1: Determine what party politician Jonas Gahr Støre belongs to
Queries:
- What party does politician Jonas Gahr Støre belong to
- Politician Jonas Gahr Støre party affiliation
- Jonas Gahr Støre political party membership

Step 2: Identify the current official positions/policies of party AP
Queries:
- Party AP official platform positions
- Current policy positions of party AP
- Party AP stance on [relevant topic from statement

Step 3: Compare the politician's statement with party positions
Queries:
- Party AP position on [specific topic from statement]
- Official party AP policy regarding [statement topic]

Respond only with the structured output as specified and every query should be in Norwegian.
"""

# ANALYSIS ---------------------------------------------------------------------------------------------------------

analysis_prompt = """
You are an expert in Norwegian political analysis who specializes in identifying inconsistencies between a political party’s actual legislative actions (votes, statements, proposals) and what the party stands for (party platform, core principles, etc.).

Your task is to analyze whether a political action or vote aligns with the party's official political line and principles.

CONTEXT (extracted from relevant documents):  
{context}

PLAN (This plan was generated to answer the question, use it to understand the context of the question before answering):
{plan}

QUESTION/ANALYSIS:  
{original_question}

GENERATED QUERIES (these queries were used in the retrieval process to gather context):
{generated_queries_from_plan}

ANALYSIS INSTRUCTIONS:  
1. Thoroughly analyze the given political action or vote  
2. Compare this action to the party’s:  
   - Political platform (party manifesto)  
   - Core principles  
   - Previous statements and positions  
   - Key policy and ideological foundation  

3. Evaluate the following factors:  
   - Is the action aligned with the party’s declared values?  
   - Does it match previous similar votes/positions?  
   - Is it consistent with the party’s overall political direction?  
   - Can any deviations be explained by special circumstances or compromises?  

4. Provide a clear judgment:  
   - TRUE if the action/vote is consistent with the party’s political line  
   - FALSE if there is a clear inconsistency or contradiction  

IMPORTANT CONSIDERATIONS:  
- Norwegian politics often involves coalitions and compromises  
- Parties may shift positions over time  
- Context and timing can influence decisions  
- Distinguish between tactical and principled decisions  

Based on the context and your expertise in Norwegian politics, provide your structured analysis of whether or not the political action aligns with the party’s official line.
Base your response on the information provided in the context and the plan and do not add any new information that is not in the context. If the context does not provide enough information to make a judgment, state that clearly.
Respond only with the structured output as specified.

IMPORTANT:
- Do not add any new information that is not in the context
- Do not make assumptions beyond what is provided in the context
- Your response should be translated to Norwegian.

You will add the relevant text chunks you have used to answer the question in the "relevant_context" field. This should be a list of strings, each string being a relevant text chunk that you have used to answer the question.
You shall not change the text chunks in any way, just add them as they are and they should be in Norwegian.
"""

remove_irrelevant_content_prompt = """
You are an expert in filtering relevant content from retrieved documents based on specific queries.
Your task is to extract only the relevant information that directly addresses the queries provided.

You receive a list of queries: {queries} and retrieved documents: {retrieved_documents} from a vector store.
You need to filter out all the non-relevant information that does not supply important information regarding the {queries}.
Your goal is to filter out the non-relevant information only.
You can remove parts of sentences that are not relevant to the queries or remove whole sentences that are not relevant to the queries.

DO NOT ADD ANY NEW INFORMATION THAT IS NOT IN THE RETRIEVED DOCUMENTS.
Output the filtered relevant content as a single string.
"""