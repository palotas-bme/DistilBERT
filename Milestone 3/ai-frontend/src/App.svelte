<script>
    import Chat from './lib/Chat.svelte';

    class QA {
        constructor(question, answer, context) {
            this.question = question;
            this.answer = answer;
            this.context = context;
        }
    }

    let qas = [new QA("What is this?", "This en example question to show the UI without the backend.", "There is no context. But something needs to be here.")];
    let question = '';
    let context;


    async function ask() {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question, context }),
        });

        if (!response.ok) {
            throw new Error('Failed to fetch response from AI');
        }

        response.json().then((result) => {
            qas.push(new QA(question, result.message, context));
            qas = qas;
            question = '';
        });
    }
</script>

<main>
    <h1>DistilBERT question answering</h1>
    <div class="main-container">
        <div class="chat-container">
            {#each qas as q}
                <div>
                    <Chat question={q.question} answer={q.answer} context={q.context} />
                </div>
            {/each}
        </div>
        <div class="question-container">
            <input placeholder="Ask something" bind:value={question} />
            <button onclick={ask} disabled={question == ''}>Ask</button>
            <textarea placeholder="context" bind:value={context}></textarea>
        </div>
    </div>
</main>

<style>
    .main-container {
        border: 1px solid rgb(19, 56, 221);
    }
    .question-container {
        border: 1px solid rgb(0, 255, 85);
        position: relative;
    }
    .chat-container {
        border: 1px solid red;
    }
</style>
