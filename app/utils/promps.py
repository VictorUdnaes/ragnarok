
MULTI_QUERY_GEN_PROMPT = """Du er en AI-språkmodellassistent. Din oppgave er å generere fem forskjellige versjoner 
        av det gitte bruker-spørsmålet for å hente relevante dokumenter fra en vektordatabank. 
        Ved å generere flere perspektiver på bruker-spørsmålet er målet ditt å hjelpe brukeren med 
        å overkomme noen av begrensningene ved avstandsbasert likhetssøk. Gi disse alternative 
        spørsmålene adskilt med nye linjer. Originalt spørsmål: {question}"""

MULTI_QUERY_FINAL_PROMPT = """Answer the following question based on this context:

            {context}

            Question: {question}
            """


        