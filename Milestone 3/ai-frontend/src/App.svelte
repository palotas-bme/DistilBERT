<script>
    import Chat from './lib/Chat.svelte';
    import Loading from './lib/Loading.svelte';

    class QA {
        constructor(question, answer, context, link) {
            this.question = question;
            this.answer = answer;
            this.context = context;
            this.link = link;
        }
    }

    let qas = [
        new QA(
            'What is this?',
            'This en example question to show the UI without the backend.',
            'There is no context. But something needs to be here.',
        ),
    ];
    let question = '';
    let context;
    let loading = false;

    async function ask() {
        loading = true;
        const orig_question = question;
        question = '';
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: orig_question, context: context }),
        });

        if (response.ok) {
            response.json().then((result) => {
                result.forEach(answer => {
                    qas.push(new QA(orig_question, answer.answer, answer.context, answer.link));
                });
                
                qas = qas;
            });
        } else {
            qas.push(new QA(orig_question, '⚠Unable to answer the question, please see the logs for details.⚠', context));
            qas = qas;
            response.text().then(console.log);
        }
        loading = false;
        const container = document.querySelector('.chat-container');
        setTimeout(() => {
            container.scrollTop = container.scrollHeight;
        }, 50);

        // For testing without backend
        // qas.push(new QA(question, "I don't know", context));
        // qas = qas;
        // const container = document.querySelector('.chat-container');
        // setTimeout(() => {
        //     container.scrollTop = container.scrollHeight;
        // }, 50);
    }
</script>

<main>
    <div class="main-container">
        <h1>DistilBERT question answering</h1>
        <div class="chat-container">
            {#each qas as q}
                <Chat question={q.question} answer={q.answer} context={q.context} link={q.link} />
            {/each}
        </div>
        <div class="question-container">
            <input placeholder="Ask something" bind:value={question} />
            {#if loading}
                <Loading />
            {:else}
                <button class="ask-button" onclick={ask} disabled={question == ''}>Ask</button>
            {/if}
            <textarea placeholder="context" bind:value={context}></textarea>
        </div>
    </div>
</main>

<style>
    .main-container {
        margin-bottom: 3rem;
        height: 90vh;
        display: flex;
        flex-direction: column;
    }
    .question-container {
        align-items: center;
    }
    .chat-container {
        overflow-y: auto;
        padding: 1rem;
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
    }

    .ask-button:disabled {
        cursor: initial;
    }
</style>
