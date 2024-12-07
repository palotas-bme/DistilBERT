<script>
    export let question = '';
    export let answer = '';
    export let text = '';
    export let link = '';

    const maxShownContext = 100;
</script>

<div class="empty"></div>
<div class="question-container">
    <div class="bubble right">
        Q: {question}
    </div>
</div>
<div class="context-container">
    {#if link != ''}
        <details>
            <summary>
                Context: {link}
            </summary>
            <p class="full-context">
                {text}
            </p>
        </details>
    {:else if text.length < maxShownContext}
        Context: {text || 'none'}
    {:else}
        <details>
            <summary>
                Context: {text.substring(0, maxShownContext)}...
            </summary>
            <p class="full-context">
                {text}
            </p>
        </details>
    {/if}
</div>
<div class="answer-container">
    <div class="bubble left">
        A: {answer}
    </div>
</div>
<div class="empty"></div>

<style>
    .question-container {
    }
    .answer-container {
    }
    .context-container {
        grid-row: span 2;
    }

    .full-context {
        z-index: 99;
        background-color: #101010;
        text-align: justify;
    }
    @media (prefers-color-scheme: light) {
        .full-context {
            background-color: #fafafa;
        }
    }

    .bubble {
        --r: 1em;
        --t: 1.5em;

        padding: 1em;
        border-inline: var(--t) solid #0000;
        border-radius: calc(var(--r) + var(--t)) / var(--r);
        mask:
            radial-gradient(100% 100% at var(--_p) 0, #0000 99%, #000 102%) var(--_p) 100% / var(--t) var(--t) no-repeat,
            linear-gradient(#000 0 0) padding-box;
        background: linear-gradient(135deg, #3607d4, #75a5c1) border-box;
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
