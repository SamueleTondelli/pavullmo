"""Prompt integrity, character accounting, publication, and Modal dispatch."""

import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from config import SyntheticGenerationConfig

try:
    spec = importlib.util.spec_from_file_location(
        "generate_synthetic_modal", Path(__file__).resolve().parents[1] / "src/synth-gen/generate_synthetic_modal.py",
    )
    generation = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = generation
    spec.loader.exec_module(generation)
except ModuleNotFoundError as error:
    if error.name != "modal":
        raise
    generation = None


class GenerationConfigTests(unittest.TestCase):
    def test_config_round_trip_and_invalid_text_settings(self):
        config = SyntheticGenerationConfig(name="trial", model="teacher", prompts="prompts.jsonl", engine={"revision": "frozen"})
        self.assertEqual(SyntheticGenerationConfig.from_toml(config.to_toml()), config)
        for settings in ({"batch_size": -1}, {"warmup_prompts": -1},
                         {"engine": {"model": "other"}},
                         {"engine": {"extra_kwargs": {"skip_tokenizer_init": True}}},
                         {"sampling": {"extra_kwargs": {"detokenize": False}}},
                         {"engine": {"extra_kwargs": {"max_num_seqs": 32}}},
                         {"engine": {"max_num_seqs": "32"}},
                         {"engine": {"gpu_memory_utilization": 1.1}},
                         {"prompt_mode": "raw", "system_prompt": "ignored"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                SyntheticGenerationConfig(name="trial", model="teacher", prompts="prompts.jsonl", **settings)

    def test_required_fields_and_dotted_overrides(self):
        values = {"name": "trial", "model": "teacher", "prompts": "prompts.jsonl"}
        for field in values:
            with self.subTest(field=field), self.assertRaises(ValueError):
                SyntheticGenerationConfig.from_dict({key: value for key, value in values.items() if key != field})
        config = SyntheticGenerationConfig.from_dict(values)
        changed = config.with_overrides({"engine.max_num_seqs": 512, "sampling.max_tokens": 2048,
                                         "chat_template_kwargs.enable_thinking": True})
        self.assertEqual(changed.engine.max_num_seqs, 512)
        self.assertEqual(changed.sampling.max_tokens, 2048)
        self.assertTrue(changed.chat_template_kwargs.enable_thinking)
        self.assertEqual(config.engine.max_num_seqs, 256)
        for override in ({"engine.max_num_seq": 10}, {"sampling.temprature": 0.1},
                         {"engine.max_num_seqs": "512"}, {"engine": {"max_num_seq": 10}}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                config.with_overrides(override)

    def test_unknown_config_is_retained_without_forwarding_to_vllm(self):
        config = SyntheticGenerationConfig.from_dict({
            "name": "trial", "model": "teacher", "prompts": "prompts.jsonl",
            "future_section": {"enabled": True},
            "engine": {"future_flag": True, "extra_kwargs": {"enable_prefix_caching": False}},
        })
        loaded = SyntheticGenerationConfig.from_toml(config.to_toml())
        self.assertTrue(loaded.engine.model_extra["future_flag"])
        self.assertTrue(loaded.model_extra["future_section"]["enabled"])
        self.assertNotIn("future_flag", loaded.engine.to_kwargs())
        self.assertFalse(loaded.engine.to_kwargs()["enable_prefix_caching"])
        self.assertNotIn("quantization", loaded.engine.to_kwargs())
        self.assertEqual(loaded.engine.quantization, "")


@unittest.skipIf(generation is None, "requires the cloud extra")
class SyntheticGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prompts = self.root / "prompts.jsonl"
        self.prompts.write_text('"Primo\\nparagrafo"\n{"prompt":"Secondo", "id":"it-2"}\n', encoding="utf-8")
        self.directory = self.root / "output"
        self.config = SyntheticGenerationConfig(name="trial", model="teacher", prompts=str(self.prompts),
                                                batch_size=1, warmup_prompts=1)

    @staticmethod
    def output(text, index=0, finish="stop"):
        return SimpleNamespace(outputs=[SimpleNamespace(
            text=text, index=index, finish_reason=finish, stop_reason=None,
        )])

    def run_job(self, llm, **kwargs):
        return generation.run_generation(
            self.config, self.prompts, self.directory,
            llm_factory=MagicMock(return_value=llm), sampling_factory=MagicMock(),
            synchronize=MagicMock(), publish=kwargs.pop("publish", MagicMock()),
            provenance={"model": "teacher"}, **kwargs,
        )

    def test_read_prompts_preserves_multiline_unicode_and_source_fields(self):
        self.assertEqual(list(generation.read_prompts(self.prompts)), [
            {"prompt": "Primo\nparagrafo"}, {"prompt": "Secondo", "id": "it-2"},
        ])
        text = self.root / "prompts.txt"
        text.write_text('  Città!  \n\nUn’altra frase\n', encoding="utf-8")
        self.assertEqual(list(generation.read_prompts(text)), [
            {"prompt": "  Città!  "}, {"prompt": "Un’altra frase"},
        ])
        for content in ('{"prompt":1}', '"   "', '{bad json}', '[]'):
            self.prompts.write_text(content)
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, ":1:"):
                list(generation.read_prompts(self.prompts))

    def test_counts_completion_characters_excludes_warmup_and_persists_all_samples(self):
        llm = MagicMock()
        two_samples = self.output("città\n")
        two_samples.outputs.extend(self.output("🙂", index=1, finish="length").outputs)
        llm.chat.side_effect = [[self.output("warmup not counted")], [two_samples], [self.output("e\u0301")]]
        publish = MagicMock()
        with patch("sys.stdout", new=io.StringIO()):
            result = self.run_job(llm, publish=publish, clock=MagicMock(side_effect=[0, 10, 12, 15, 18, 25]))
        # Unicode code points, not UTF-8 bytes or prompt/JSON framing characters.
        self.assertEqual(result["generated_chars"], 9)
        self.assertEqual(result["generated_sequences"], 3)
        self.assertEqual(result["length_limited_sequences"], 1)
        self.assertEqual(result["processed_prompts"], 2)
        self.assertEqual(result["generation_seconds"], 5)
        self.assertEqual(result["chars_per_second"], 1.8)
        self.assertEqual(result["job_chars_per_second"], 9 / 25)
        self.assertEqual(result["status"], "completed")
        rows = [json.loads(line) for line in (self.directory / "generations.jsonl").read_text().splitlines()]
        self.assertEqual([row["text"] for row in rows], ["città\n", "🙂", "e\u0301"])
        self.assertEqual([(row["prompt_index"], row["sample_index"]) for row in rows], [(0, 0), (0, 1), (1, 0)])
        self.assertEqual(rows[-1]["source"]["id"], "it-2")
        self.assertEqual(json.loads((self.directory / "metrics.json").read_text()), result)
        self.assertFalse((self.directory / "generations.jsonl.partial").exists())
        self.assertEqual(publish.call_count, 4)
        self.assertEqual(llm.chat.call_args.kwargs["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(llm.chat.call_args_list[1].args[0][0][-1]["content"], "Primo\nparagrafo")

    def test_failed_batch_retains_completed_text_and_commits_failure(self):
        llm = MagicMock()
        llm.chat.side_effect = [[self.output("warmup")], [self.output("già salvato")], RuntimeError("inference failed")]
        publish = MagicMock()
        with patch("sys.stdout", new=io.StringIO()), self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.run_job(llm, publish=publish)
        rows = [json.loads(line) for line in (self.directory / "generations.jsonl.partial").read_text().splitlines()]
        self.assertEqual(rows[0]["text"], "già salvato")
        metrics = json.loads((self.directory / "metrics.json").read_text())
        self.assertEqual(metrics["status"], "failed")
        self.assertEqual(metrics["processed_prompts"], 1)
        self.assertNotIn("chars_per_second", metrics)
        self.assertEqual(publish.call_count, 2)

    def test_raw_completion_and_model_load_failure(self):
        self.config = self.config.with_overrides({"prompt_mode": "raw", "warmup_prompts": 0})
        llm = MagicMock()
        llm.generate.side_effect = [[self.output("uno")], [self.output("due")]]
        with patch("sys.stdout", new=io.StringIO()):
            self.run_job(llm)
        llm.chat.assert_not_called()
        self.assertEqual(llm.generate.call_args_list[0].args[0], ["Primo\nparagrafo"])
        with self.assertRaises(FileExistsError):
            self.run_job(llm)
        self.directory = self.root / "failed-load"
        publish = MagicMock()
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            generation.run_generation(
                self.config, self.prompts, self.directory,
                llm_factory=MagicMock(side_effect=RuntimeError("load failed")),
                sampling_factory=MagicMock(), synchronize=MagicMock(), publish=publish,
                provenance={},
            )
        self.assertEqual(json.loads((self.directory / "metrics.json").read_text())["status"], "failed")
        publish.assert_called_once()

    def test_resolved_toml_preserves_native_options_and_is_standalone(self):
        self.config = self.config.with_overrides({"warmup_prompts": 0})
        self.config.engine.extra_kwargs["enable_prefix_caching"] = False
        self.config.sampling.extra_kwargs["repetition_penalty"] = 1.05
        llm = MagicMock()
        llm.chat.side_effect = [[self.output("uno")], [self.output("due")]]
        with patch("sys.stdout", new=io.StringIO()):
            self.run_job(llm)
        saved = tomllib.loads((self.directory / "config.toml").read_text())
        self.assertFalse(saved["engine"]["extra_kwargs"]["enable_prefix_caching"])
        self.assertEqual(saved["sampling"]["extra_kwargs"]["repetition_penalty"], 1.05)
        self.assertNotIn("extends", saved)
        self.assertEqual(SyntheticGenerationConfig.load(self.directory / "config.toml"), self.config)
        self.assertFalse((self.directory / "config.json").exists())

    def test_missing_completions_are_a_failure(self):
        self.config = self.config.with_overrides({"warmup_prompts": 0})
        llm = MagicMock()
        llm.chat.return_value = []
        with self.assertRaisesRegex(RuntimeError, "every prompt"):
            self.run_job(llm)

    def test_dispatch_uploads_prompts_forwards_native_options_and_returns_metrics(self):
        (self.root / "base.toml").write_text('model="teacher"\n[engine]\nmax_num_seqs=64\n')
        (self.root / "first.toml").write_text('[engine]\nmax_num_seqs=128\nquantization="fp8"\n')
        (self.root / "second.toml").write_text('[engine]\nmax_num_seqs=256\nrevision="pinned"\n')
        source = self.root / "run.toml"
        source.write_text('extends=["base.toml"]\n' + self.config.to_toml())
        original = source.read_text()
        outputs, cache = MagicMock(), MagicMock()
        function = MagicMock()
        function.remote.return_value = {"chars_per_second": 5000, "job_chars_per_second": 4000}
        captured = []
        def capture(path, remote_path):
            captured.append((Path(path).read_text(), remote_path))
        outputs.batch_upload.return_value.__enter__.return_value.put_file.side_effect = capture
        with patch.object(generation.modal.Volume, "from_name", side_effect=[outputs, cache]), \
                patch.object(generation.generate_remote, "with_options", return_value=function) as options, \
                patch("sys.stdout", new=io.StringIO()):
            result = generation.main(
                config=str(source.with_suffix("")), gpu="H100", extends="first;second.toml",
                override="engine.max_num_seqs=512;engine.quantization=fp8;engine.revision=pinned;"
                         "sampling.max_tokens=2048;sampling.n=2;batch_size=8192;warmup_prompts=0",
            )
        self.assertEqual(result, function.remote.return_value)
        config_toml, volume, provenance = function.remote.call_args.args
        data = tomllib.loads(config_toml)
        self.assertNotIn("extends", data)
        self.assertEqual(source.read_text(), original)
        self.assertEqual((data["name"], volume, provenance["prompt_count"]), ("trial", "pavullmo-training", 2))
        self.assertEqual(data["engine"]["max_num_seqs"], 512)
        self.assertEqual(data["engine"]["performance_mode"], "throughput")
        self.assertEqual(data["engine"]["revision"], "pinned")
        self.assertEqual(data["sampling"]["n"], 2)
        self.assertEqual(data["batch_size"], 8192)
        self.assertEqual(captured[0][1], "/synthetic/trial/prompts.jsonl")
        self.assertEqual(json.loads(captured[0][0].splitlines()[0])["prompt"], "Primo\nparagrafo")
        self.assertEqual(options.call_args.kwargs["volumes"], {"/outputs": outputs, "/cache": cache})

    def test_invalid_local_input_never_creates_cloud_resources(self):
        source = self.root / "run.toml"
        self.config.save(source)
        for kwargs in ({"override": "name=../bad"}, {"override": "engine.max_num_seq=10"},
                       {"override": "engine={model='other'}"},
                       {"override": "batch_size=-3"}, {"override": "warmup_prompts=-2"},
                       {"override": "engine.tensor_parallel_size=2"},
                       {"timeout_seconds": 90000}):
            with self.subTest(kwargs=kwargs), patch.object(generation.modal.Volume, "from_name") as volumes:
                with self.assertRaises(ValueError):
                    generation.main(config=str(source), **kwargs)
                volumes.assert_not_called()
        for text in ('""', "", '{"prompt":false}'):
            self.prompts.write_text(text)
            with patch.object(generation.modal.Volume, "from_name") as volumes, self.assertRaises(ValueError):
                generation.main(config=str(source))
            volumes.assert_not_called()

    def test_recursive_composition_cli_search_and_saved_config_independence(self):
        (self.root / "nested").mkdir()
        (self.root / "nested" / "base.toml").write_text('model="teacher"\n[engine]\nmax_num_seqs=64\n')
        (self.root / "base.toml").write_text('extends=["nested/base.toml"]\n[sampling]\ntop_p=0.8\n')
        (self.root / "extra.toml").write_text('[engine]\nmax_num_seqs=512\n[sampling]\ntop_p=0.7\n')
        (self.root / "last.toml").write_text('[sampling]\ntop_p=0.6\n')
        source = self.root / "run.toml"
        source.write_text('extends=["base.toml"]\nname="composed"\nprompts=' + json.dumps(str(self.prompts))
                          + '\n[engine]\nmax_num_seqs=256\n')
        worker = MagicMock()
        worker.remote.return_value = {"chars_per_second": 100, "job_chars_per_second": 50}
        with patch.object(generation.modal.Volume, "from_name"), \
                patch.object(generation.generate_remote, "with_options", return_value=worker), \
                patch("sys.stdout", new=io.StringIO()):
            generation.main(config=str(source), extends="extra;last", override="sampling.max_tokens=4096")
        saved = worker.remote.call_args.args[0]
        run = SyntheticGenerationConfig.from_toml(saved)
        self.assertEqual(run.engine.max_num_seqs, 256)  # Main file wins over CLI includes.
        self.assertEqual(run.sampling.top_p, 0.6)  # Later CLI includes win over earlier ones.
        self.assertEqual(run.sampling.max_tokens, 4096)
        self.assertEqual(run.model, "teacher")
        for path in self.root.rglob("*.toml"):
            path.unlink()
        self.assertEqual(SyntheticGenerationConfig.from_toml(saved), run)

    def test_bundled_config_and_cli_include_resolve_from_synthetic_directory(self):
        worker = MagicMock()
        worker.remote.return_value = {"chars_per_second": 100, "job_chars_per_second": 50}
        with patch.object(generation.modal.Volume, "from_name"), \
                patch.object(generation.generate_remote, "with_options", return_value=worker), \
                patch("sys.stdout", new=io.StringIO()):
            generation.main(config="synthetic_qwen35_4b", override=f'name=bundled;prompts="{self.prompts}"')
            run = SyntheticGenerationConfig.from_toml(worker.remote.call_args.args[0])
            self.assertEqual(run.model, "Qwen/Qwen3.5-4B")
            self.assertEqual(run.sampling.top_k, 20)
            source = self.root / "run.toml"
            source.write_text(f'name="includes"\nmodel="teacher"\nprompts="{self.prompts}"\n')
            generation.main(config=str(source), extends="synthetic_qwen35_4b")
            run = SyntheticGenerationConfig.from_toml(worker.remote.call_args.args[0])
            self.assertEqual(run.model, "teacher")
            self.assertEqual(run.sampling.top_k, 20)

    def test_remote_accepts_resolved_toml_without_loading_presets(self):
        fake_torch = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda="13.0"),
                                     cuda=SimpleNamespace(synchronize=MagicMock(), get_device_name=lambda: "test-gpu"))
        fake_vllm = SimpleNamespace(__version__="test", LLM=MagicMock(), SamplingParams=MagicMock())
        with patch.dict("sys.modules", torch=fake_torch, vllm=fake_vllm), \
                patch.object(generation.modal.Volume, "from_name"), \
                patch.object(generation, "run_generation", return_value={"completed": True}) as run:
            result = generation.generate_remote.local(self.config.to_toml(), "pavullmo-training", {})
        self.assertEqual(result, {"completed": True})
        self.assertEqual(run.call_args.args[0], self.config)
        self.assertEqual(run.call_args.args[1], Path("/outputs/synthetic/trial/prompts.jsonl"))

    def test_remote_import_does_not_construct_local_image(self):
        spec = importlib.util.spec_from_file_location("remote_synthetic", generation.__file__)
        module = importlib.util.module_from_spec(spec)
        module.__file__ = "/root/generate_synthetic_modal.py"
        with patch.object(generation.modal, "is_local", return_value=False), \
                patch.object(generation.modal.Image, "from_registry") as registry, \
                patch.object(generation.modal.Image, "uv_pip_install") as install:
            spec.loader.exec_module(module)
        registry.assert_not_called()
        install.assert_not_called()


if __name__ == "__main__":
    unittest.main()
