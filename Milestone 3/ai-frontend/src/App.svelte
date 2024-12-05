<script>
    import Chat from './lib/Chat.svelte';

    class QA {
        constructor(question, answer, context) {
            this.question = question;
            this.answer = answer;
            this.context = context;
        }
    }

    let qas = [];
    let inputQuestion;
    let inputContext;

    function ask() {

        // TODO Get answer form api
        qas.push(new QA(inputQuestion, "I don't know", inputContext));
        qas = qas;
        inputQuestion = '';
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
            <input placeholder="Ask something" bind:value={inputQuestion} />
            <button onclick={ask} disabled={inputQuestion == ""}>Ask</button>
            <textarea placeholder="context" bind:value={inputContext}></textarea>
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
