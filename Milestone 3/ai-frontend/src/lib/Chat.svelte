<script>
    export let question = '';
    export let answer = '';
    export let context = '';

    const maxShownContext = 10;
</script>

<div class="qa-container">
    <div class="answer-container">
        <div class="bubble left">
            A: {answer}
        </div>
    </div>
    <div class="question-container">
        <div class="bubble right">
            Q: {question}
        </div>
    </div>
    <div class="context-container">
        Context: {context.substring(0, maxShownContext)}
        {#if context.length > maxShownContext}
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
        padding: 3em;
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
        flex-direction: row;
        flex-grow: 1;
        align-items: flex-end;
        max-width: 33%;
    }
    .context-container {
        min-height: 2rem;
        flex-grow: 2;
        max-width: 33%;
    }

    .context-container:hover > .full-context {
        display: block;
    }

    .full-context {
        display: none;
        position: absolute;
        z-index: 99;
    }

    .bubble {
        --r: 1em; /* the radius */
        --t: 1.5em; /* the size of the tail */

        max-width: 300px;
        padding: 1em;
        border-inline: var(--t) solid #0000;
        border-radius: calc(var(--r) + var(--t)) / var(--r);
        mask:
            radial-gradient(100% 100% at var(--_p) 0, #0000 99%, #000 102%) var(--_p) 100% / var(--t) var(--t) no-repeat,
            linear-gradient(#000 0 0) padding-box;
        background: linear-gradient(135deg, #3701ea, #75a5c1) border-box;
        color: #fff;
    }

    .left {
        --_p: 0;
        border-bottom-left-radius: 0 0;
        place-self: start;
    }
    .right {
        --_p: 100%;
        border-bottom-right-radius: 0 0;
        place-self: end;
    }
</style>
