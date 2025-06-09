<svelte:options accessors={true} />

<script lang="ts">
    import type { Gradio } from "@gradio/utils";
    import { BlockTitle } from "@gradio/atoms";
    import { Block } from "@gradio/atoms";
    import { StatusTracker } from "@gradio/statustracker";
    import type { LoadingStatus } from "@gradio/statustracker";

    // export let gradio: Gradio<{}>;
    export let label = "Agent Thoughts";
    export let elem_id = "";
    export let elem_classes: string[] = [];
    export let visible = true;
    export let value: string[] = [];  // Now an array!
    export let show_label: boolean;
    export let scale: number | null = null;
    export let min_width: number | undefined = undefined;
    // export let loading_status: LoadingStatus | undefined = undefined;
    export let root: string;
</script>

<Block
    {visible}
    {elem_id}
    {elem_classes}
    {scale}
    {min_width}
    allow_overflow={true}
    padding={true}
>
    <!-- {#if loading_status}
        <StatusTracker
            autoscroll={gradio.autoscroll}
            i18n={gradio.i18n}
            {...loading_status}
            on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
        />
    {/if} -->

    <div class="container">
        <BlockTitle {root} {show_label} info={undefined}>{label}</BlockTitle>
        <ul class="thought-list">
            {#each value as thought, idx (idx)}
                <li class="thought-item">{thought}</li>
            {/each}
        </ul>
    </div>
</Block>

<style>
    .thought-list {
        list-style: none;
        margin: 0;
        padding: 0;
        max-height: 300px;
        overflow-y: auto;
        background: var(--input-background-fill);
        border: var(--input-border-width) solid var(--input-border-color);
        border-radius: var(--input-radius);
        font-size: var(--input-text-size);
    }
    .thought-item {
        padding: 0.75em 1em;
        border-bottom: 1px solid var(--input-border-color);
    }
    .thought-item:last-child {
        border-bottom: none;
    }
</style>
