from sql import NL2SQL
from sql import SchemaRetriever
from classifier import run_pipeline
from rich import print_json

engine = NL2SQL()

query = "What is the password that user rajit uses?"

relevant_tables_json = {
    "faculty": {
        "columns": [
            {
                "column_name": "faculty_id",
                "data_type": "integer",
                
            },
            {
                "column_name": "first_name",
                "data_type": "character varying",
                
            },
            {
                "column_name": "last_name",
                "data_type": "character varying",
                
            },
            {
                "column_name": "email",
                "data_type": "character varying",
                
            },
            {
                "column_name": "phone",
                "data_type": "character varying",
                
            },
            {
                "column_name": "department_id",
                "data_type": "integer",
                
            }
        ],
        "primary_keys": [
            "faculty_id"
        ],
        "foreign_keys": [
            {
                "column": "department_id",
                "references_table": "departments",
                "references_column": "department_id"
            }
        ],
        "sample_rows": [
            {
                "faculty_id": 1,
                "first_name": "Amit",
                "last_name": "Sharma",
                "email": "amit@college.com",
                "phone": "9123456701",
                "department_id": 1
            },
            {
                "faculty_id": 2,
                "first_name": "Neha",
                "last_name": "Kulkarni",
                "email": "neha@college.com",
                "phone": "9123456702",
                "department_id": 1
            }
        ]
    },
       
}

retriever = SchemaRetriever()
# schema = retriever.retrieve(query, top_k=5)


while True:
    query = input("\nEnter your query (type 'exit' to stop): ")
    if query.lower() == "exit":
        print("Stopped.")
        break
    resolved = run_pipeline(query)
    # This is for extracting only one table
    schema = retriever.retrieve(resolved, top_k=5)

    print_json(data=schema)
    sql = engine.generate(resolved, schema)
    print("[SQL]")
    print(sql)

