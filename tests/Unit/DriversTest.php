<?php

declare(strict_types=1);

use Laravel\Ai\AnonymousAgent;
use Laravel\Ai\Embeddings;
use Laravel\Ai\Prompts\AgentPrompt;
use Laravel\Ai\Prompts\EmbeddingsPrompt;
use Laravel\Ai\Responses\AgentResponse;
use Laravel\Ai\Responses\Data\Meta;
use Laravel\Ai\Responses\Data\TextUsage;
use Laravel\Ai\Responses\Data\Usage;
use Laravel\Ai\Responses\EmbeddingsResponse;
use Pest\Evals\Drivers\LaravelAiEmbeddings;
use Pest\Evals\Drivers\LaravelAiJudge;
use Pest\Evals\Tests\TestCase;

uses(TestCase::class);

afterEach(function (): void {
    putenv('PEST_EVALS_LARAVEL_SCORING_PROVIDER');
    putenv('PEST_EVALS_LARAVEL_SCORING_MODEL');
    putenv('PEST_EVALS_LARAVEL_EMBEDDING_PROVIDER');
    putenv('PEST_EVALS_LARAVEL_EMBEDDING_MODEL');
});

describe('LaravelAiJudge', function (): void {
    it('uses the hardcoded defaults when nothing is provided', function (): void {
        $judge = new LaravelAiJudge();

        expect($judge->provider)->toBe('openai')
            ->and($judge->model)->toBe('gpt-5.4-nano');
    });

    it('accepts an explicit provider and model', function (): void {
        $judge = new LaravelAiJudge(provider: 'anthropic', model: 'claude');

        expect($judge->provider)->toBe('anthropic')
            ->and($judge->model)->toBe('claude');
    });

    it('falls back to environment variables', function (): void {
        putenv('PEST_EVALS_LARAVEL_SCORING_PROVIDER=azure');
        putenv('PEST_EVALS_LARAVEL_SCORING_MODEL=gpt-env');

        $judge = new LaravelAiJudge();

        expect($judge->provider)->toBe('azure')
            ->and($judge->model)->toBe('gpt-env');
    });

    it('prefers an explicit value over an env var', function (): void {
        putenv('PEST_EVALS_LARAVEL_SCORING_MODEL=gpt-env');

        expect(new LaravelAiJudge(model: 'gpt-explicit')->model)->toBe('gpt-explicit');
    });

    it('returns the SDK response text and forwards the judge request', function (): void {
        AnonymousAgent::fake([
            new AgentResponse('judge-test', '{"score":0.7,"reasoning":"Relevant"}', new TextUsage(
                inputTokens: 120,
                outputTokens: 35,
                cacheReadInputTokens: 20,
                cacheWriteInputTokens: 10,
                reasoningTokens: 15,
            ), new Meta('anthropic', 'claude-test')),
        ])->preventStrayPrompts();

        $result = new LaravelAiJudge(provider: 'anthropic', model: 'claude-test')
            ->generate('Evaluate relevance.', 'Question: refund? Answer: 30 days.');

        expect($result)->toBe('{"score":0.7,"reasoning":"Relevant"}');

        AnonymousAgent::assertPrompted(fn (AgentPrompt $prompt): bool => $prompt->agent->instructions() === 'Evaluate relevance.'
            && $prompt->prompt === 'Question: refund? Answer: 30 days.'
            && $prompt->provider->name() === 'anthropic'
            && $prompt->model === 'claude-test'
        );
    });
});

describe('LaravelAiEmbeddings', function (): void {
    it('uses the hardcoded defaults when nothing is provided', function (): void {
        $embeddings = new LaravelAiEmbeddings();

        expect($embeddings->provider)->toBe('openai')
            ->and($embeddings->model)->toBe('text-embedding-3-small');
    });

    it('accepts an explicit provider and model', function (): void {
        $embeddings = new LaravelAiEmbeddings(provider: 'voyage', model: 'voyage-3');

        expect($embeddings->provider)->toBe('voyage')
            ->and($embeddings->model)->toBe('voyage-3');
    });

    it('falls back to environment variables', function (): void {
        putenv('PEST_EVALS_LARAVEL_EMBEDDING_PROVIDER=cohere');
        putenv('PEST_EVALS_LARAVEL_EMBEDDING_MODEL=embed-env');

        $embeddings = new LaravelAiEmbeddings();

        expect($embeddings->provider)->toBe('cohere')
            ->and($embeddings->model)->toBe('embed-env');
    });

    it('returns ordered SDK vectors independently of the usage payload', function (): void {
        Embeddings::fake([
            new EmbeddingsResponse([[2 => 0.25, 7 => -0.5], [4 => 0.75, 9 => 0.125]], new Usage(inputTokens: 17), new Meta('cohere', 'embed-test')),
        ])->preventStrayEmbeddings();

        $result = new LaravelAiEmbeddings(provider: 'cohere', model: 'embed-test')
            ->embed(['actual response', 'reference answer']);

        expect($result)->toBe([[0.25, -0.5], [0.75, 0.125]]);

        Embeddings::assertGenerated(fn (EmbeddingsPrompt $prompt): bool => $prompt->inputs === ['actual response', 'reference answer']
            && $prompt->provider->name() === 'cohere'
            && $prompt->model === 'embed-test'
        );
    });
});
