<script>
    let rating;
	export let question;
	export let answer;
	export let text;

    function rate(e) {
        rating = e.target.value;
		console.log(question, answer, text)
    }
</script>

{#if rating == null}
    <div>
        Rate this answer:
        <span>
            <button class="star rating" onclick={rate} value="1">★</button>
            <button class="star rating" onclick={rate} value="2">★</button>
            <button class="star rating" onclick={rate} value="3">★</button>
            <button class="star rating" onclick={rate} value="4">★</button>
            <button class="star rating" onclick={rate} value="5">★</button>
        </span>
    </div>
{:else}
    Rating:
    {#each { length: rating } as _, i}
        <button class="star rated">★</button>
    {/each}
{/if}

<style>
    .rating {
        font-weight: 500;
        font-size: 1em;
    }
    .rating:hover {
        color: red;
    }
    .rating:has(~ .star:hover) {
        color: red;
    }

    button,
    .star {
        padding: 0;
        margin: 0;
		cursor: pointer;
    }
    button:active,
    .star {
        background: initial;
    }

    button,
    .rated {
        cursor: initial;
    }
</style>
