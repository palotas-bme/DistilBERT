<script>
    export let question = '';
    export let answer = '';
    export let context = '';
</script>

<div class="qa-container">
    <div class="answer-container">A: {answer}</div>
    <div class="question-container">Q: {question}</div>
    <div class="context-container">Context: {context.substring(0,100)}
        {#if (context.length > 10)}
        ...
        <div class="full-context">
            {context}
        </div>
        {/if}
    </div>
</div>

<style>
    .qa-container {
        display: flex;
        min-height: 3rem;
    }
    .question-container {
        min-height: 1rem;
        flex-grow: 1;
        max-width: 33%;
    }
    .answer-container {
        min-height: 3rem;
        display: flex;
        flex-direction: row;;
        flex-grow: 1;
        align-items: flex-end;
        max-width: 33%;

    }
    .context-container {
        min-height: 2rem;
        flex-grow: 2;
        max-width: 33%;
    }

    .context-container:hover > .full-context{
        display: block;
    }

    .full-context{
        display: none;
        position: absolute;
        z-index: 99;
    }
</style>
