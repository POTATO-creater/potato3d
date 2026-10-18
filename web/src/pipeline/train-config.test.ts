// Masked training options.  pnpm test:web
import assert from "node:assert/strict";
import { test } from "node:test";
import type { TrainOptions } from "../../../src/lib/api.ts";
import { PRESETS, REDUCE_HAZE_WEIGHT } from "../../../src/lib/presets.ts";
import { trainConfig } from "./train-config.ts";

const opts: TrainOptions = {
  totalSteps: 15000,
  shDegree: 1,
  maxSplats: 2_000_000,
  maxResolution: 1920,
  exportEvery: 2500,
  evalSplitEvery: 0,
  ssimWeight: 0,
  opacBinaryWeight: 0.01,
  depthDistortionWeight: REDUCE_HAZE_WEIGHT,
};

test("train config: random background and depth distortion only with masks", () => {
  const plain = trainConfig(opts, false);
  assert.equal(plain["total-train-iters"], 15000);
  assert.equal(plain["opac-binary-weight"], 0.01);
  for (const k of Object.keys(plain)) assert.ok(!/background|distortion|geom-reg|eval-every/.test(k), k);

  const masked = trainConfig(opts, true);
  assert.deepEqual(masked["background-color"], [0.5, 0.5, 0.5]);
  assert.equal(masked["background-noise-strength"], 0.5);
  assert.equal(masked["depth-distortion-weight"], REDUCE_HAZE_WEIGHT);
  assert.equal(masked["geom-reg-start"], 0.25);
  assert.equal(masked["geom-reg-every"], 2);
  assert.equal(masked["geom-reg-downscale"], 1);
  assert.equal(masked["geom-reg-crop"], true);
});

test("train config: no distortion keys when the weight is 0 or missing", () => {
  for (const depthDistortionWeight of [0, undefined]) {
    const cfg = trainConfig({ ...opts, depthDistortionWeight }, true);
    for (const k of Object.keys(cfg)) assert.ok(!/distortion|geom-reg/.test(k), k);
  }
});

test("presets: reduce haze is off in every preset", () => {
  for (const p of PRESETS) assert.equal(p.settings.train.depthDistortionWeight, undefined, p.id);
});

test("train config: the removed scaleLossWeight of older settings is ignored", () => {
  const cfg = trainConfig({ ...opts, scaleLossWeight: 1e-6 } as TrainOptions, true);
  for (const k of Object.keys(cfg)) assert.ok(!/scale-loss/.test(k), k);
});
