import assert from "node:assert/strict";
import { test } from "node:test";
import { groupSources } from "./sources.ts";

const IMG = ["jpg", "jpeg", "png", "webp"];

test("videos, packages and folders become one input each; loose images group by folder", () => {
  const got = groupSources(
    [
      "/a/orbit.mp4",
      "/b/frames/0001.png",
      "/a/cat.MOV",
      "/b/frames/0002.png",
      "/c/shots",
      "/d/saved.gorbit",
      "/e/0001.jpg",
      "/f/take.v2",
    ],
    IMG,
    "gorbit",
  );
  assert.deepEqual(got, [
    { kind: "video", path: "/a/orbit.mp4" },
    { kind: "images", paths: ["/b/frames/0001.png", "/b/frames/0002.png"] },
    { kind: "video", path: "/a/cat.MOV" },
    { kind: "images", paths: ["/c/shots"] },
    { kind: "package", path: "/d/saved.gorbit" },
    { kind: "images", paths: ["/e/0001.jpg"] },
    { kind: "images", paths: ["/f/take.v2"] },
  ]);
});

test("Windows paths", () => {
  assert.deepEqual(groupSources(["C:\\x\\f\\1.jpg", "C:\\x\\f\\2.jpg", "C:\\x\\g"], IMG, "gorbit"), [
    { kind: "images", paths: ["C:\\x\\f\\1.jpg", "C:\\x\\f\\2.jpg"] },
    { kind: "images", paths: ["C:\\x\\g"] },
  ]);
});
