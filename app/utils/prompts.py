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

# ANALYSIS ---------------------------------------------------------------------------------------------------------

analysis_prompt = """
You are an expert in Norwegian political analysis who specializes in identifying inconsistencies between a political party’s actual legislative actions (votes, statements, proposals) and what the party stands for (party platform, core principles, etc.).

Your task is to analyze whether a political action or vote aligns with the party's official political line and principles.

CONTEXT (extracted from relevant documents):  
{context}

QUESTION/ANALYSIS:  
{question}

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

Based on the context and your expertise in Norwegian politics, provide your structured analysis of whether the political action aligns with the party’s official line.

Respond only with the structured output as specified.
"""