import gradio as gr
import requests

API_URL = "http://localhost:8000/generate-sql"

# Realistic schema example for the demo
schema_example = """CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    name VARCHAR(100),
    plan_type VARCHAR(20),
    created_at DATETIME
);

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    amount DECIMAL(10, 2),
    category VARCHAR(50),
    created_at DATETIME,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE refunds (
    refund_id INT PRIMARY KEY,
    order_id INT,
    reason VARCHAR(100),
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);"""

question_example = "What is the total revenue by category for premium customers who signed up in 2023?"

def predict(question, schema):
    try:
        res = requests.post(API_URL, json={"question": question, "schema": schema}, timeout=60)
        if res.status_code == 200:
            data = res.json()
            sql = data["sql"]
            latency = data["latency_ms"] / 1000
            cache = "⚡ Served from Redis Cache" if data.get("from_cache") else f"🤖 Model Generation Time: {latency:.2f} seconds"
            return sql, cache
        return f"-- Error {res.status_code} --\n{res.text}", "Error"
    except Exception as e:
        return f"-- Connection Error --\n{str(e)}", "❌ Cannot connect to FastAPI server. Ensure it is running on port 8000."

with gr.Blocks(theme=gr.themes.Soft(), title="Grounded-SQL") as demo:
    gr.Markdown("# 🚀 Grounded-SQL: Natural Language to SQL")
    gr.Markdown("Interactive demo powered by **Mistral-7B (QLoRA r=64)** with INT8 quantization. Type a question and provide the database schema to generate the SQL query.")
    
    with gr.Row():
        with gr.Column(scale=1):
            schema_in = gr.Code(label="Database Schema (DDL)", language="sql", value=schema_example, lines=20)
        
        with gr.Column(scale=1):
            question_in = gr.Textbox(label="Natural Language Question", value=question_example, lines=3)
            submit_btn = gr.Button("Generate SQL", variant="primary")
            
            gr.Markdown("### Output")
            status_out = gr.Markdown("*Ready*")
            sql_out = gr.Code(label="Generated SQL", language="sql", lines=8)
            
    submit_btn.click(fn=predict, inputs=[question_in, schema_in], outputs=[sql_out, status_out])

if __name__ == "__main__":
    print("Starting Gradio public server...")
    # share=True creates a public https://xxxx.gradio.live link!
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
