#!/usr/bin/env python
"""探针 B 视觉特征抽取：AMI 近景视频 → FaceLandmarker → 24 维/帧 → 池化。

每个事件窗口 [pause_start - lookback, pause_end]（因果，默认 1.6s 前视回看）。
24 维 = 21 blendshape(口/眼/注视/眉) + 3 头姿(yaw/pitch/roll)。
池化 = mean(24)+std(24)+last_valid(24)+valid_ratio(1) = 73 维/事件。

用法：python probe_b_visual_extract.py ES2002a [ES2002b ...] [--lookback 1.6]
输入 results/probe_b/<m>_labeled.json + data/AMI/media/<m>/<m>.<cam>.avi
输出 results/probe_b/<m>_visual.npz (X[N,73], y[N], dur[N], spk[N])
"""
import sys, os, json, argparse, html
import numpy as np, cv2
import xml.etree.ElementTree as ET
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

ANNO = "data/AMI/annotations/unzipped"
# 资产不入库（见根 .gitignore *.task）；下载：
#   curl -L -o AV-SemanticVAD/models/face_landmarker.task \
#     https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
MODEL = "AV-SemanticVAD/models/face_landmarker.task"
BLEND = [  # 21 个 blendshape（口6/注视8/眨眼2/眉5）
 "jawOpen","mouthClose","mouthPucker","mouthFunnel","mouthStretchLeft","mouthStretchRight",
 "eyeLookUpLeft","eyeLookDownLeft","eyeLookInLeft","eyeLookOutLeft",
 "eyeLookUpRight","eyeLookDownRight","eyeLookInRight","eyeLookOutRight",
 "eyeBlinkLeft","eyeBlinkRight",
 "browDownLeft","browDownRight","browInnerUp","browOuterUpLeft","browOuterUpRight"]
NDIM = len(BLEND) + 3  # +yaw/pitch/roll = 24

def spk2cam(meeting):
    root = ET.parse(f"{ANNO}/corpusResources/meetings.xml").getroot()
    m = {}
    for meet in root.iter():
        if meet.tag.split('}')[-1] == 'meeting' and meet.get('observation') == meeting:
            for sp in meet:
                m[sp.get('nxt_agent')] = sp.get('camera')
    return m

def euler_from_matrix(T):
    R = np.array(T)[:3, :3]
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        x = np.arctan2(R[2,1], R[2,2]); y = np.arctan2(-R[2,0], sy); z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1]); y = np.arctan2(-R[2,0], sy); z = 0.0
    return np.array([y, x, z]) / np.pi  # yaw,pitch,roll 归一化

def frame_feat(res):
    if not res.face_blendshapes:
        return None
    d = {c.category_name: c.score for c in res.face_blendshapes[0]}
    bs = np.array([d.get(n, 0.0) for n in BLEND], dtype=np.float32)
    head = euler_from_matrix(res.facial_transformation_matrixes[0]) if res.facial_transformation_matrixes else np.zeros(3)
    return np.concatenate([bs, head.astype(np.float32)])

def extract_meeting(meeting, lookback):
    labeled = json.load(open(f"AV-SemanticVAD/results/probe_b/{meeting}_labeled.json"))
    cam = spk2cam(meeting)
    # 每个 camera 建一个 landmarker + VideoCapture（按需）
    base = mpp.BaseOptions(model_asset_path=MODEL)
    opt = vision.FaceLandmarkerOptions(base_options=base, output_face_blendshapes=True,
            output_facial_transformation_matrixes=True, num_faces=1,
            running_mode=vision.RunningMode.VIDEO)
    caps = {}
    def get_cap(c):
        if c not in caps:
            p = f"data/AMI/media/{meeting}/{meeting}.{c}.avi"
            caps[c] = (cv2.VideoCapture(p), p)
        return caps[c]

    X, y, durs, spks, vr = [], [], [], [], []
    seqs = []  # 每事件的逐帧序列 [T,25]（24特征+1有效位），供 GRU 用
    for e in labeled:
        c = cam.get(e['spk'])
        cap, path = get_cap(c)
        if not cap.isOpened():
            print(f"  !! missing video {path}"); continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        t0 = max(0.0, e['pause_start'] - lookback); t1 = e['pause_end']
        f0 = int(t0 * fps); f1 = int(t1 * fps)
        lm = vision.FaceLandmarker.create_from_options(opt)  # 每事件新建，避免 VIDEO 时间戳倒退
        feats = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        for fi in range(f0, f1 + 1):
            ok, fr = cap.read()
            if not ok: break
            mimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            res = lm.detect_for_video(mimg, int(fi / fps * 1000))
            ff = frame_feat(res)
            feats.append(ff if ff is not None else None)
        lm.close()
        # 逐帧序列：无效帧置零 + 有效位
        seq = np.array([np.concatenate([f, [1.0]]) if f is not None
                        else np.zeros(NDIM + 1, np.float32) for f in feats],
                       dtype=np.float32) if feats else np.zeros((1, NDIM + 1), np.float32)
        seqs.append(seq)
        valid = [f for f in feats if f is not None]
        ratio = len(valid) / max(1, len(feats))
        if valid:
            V = np.stack(valid)
            mean = V.mean(0); std = V.std(0); last = valid[-1]
        else:
            mean = std = last = np.zeros(NDIM, np.float32)
        X.append(np.concatenate([mean, std, last, [ratio]]).astype(np.float32))
        y.append(e['ci']); durs.append(e['dur']); spks.append(e['spk']); vr.append(ratio)
    for cap, _ in caps.values():
        cap.release()
    return np.array(X), np.array(y), np.array(durs), np.array(spks), np.array(vr), seqs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meetings", nargs="+")
    ap.add_argument("--lookback", type=float, default=1.6)
    args = ap.parse_args()
    for m in args.meetings:
        X, y, durs, spks, vr, seqs = extract_meeting(m, args.lookback)
        outf = f"AV-SemanticVAD/results/probe_b/{m}_visual.npz"
        np.savez(outf, X=X, y=y, dur=durs, spk=spks, valid_ratio=vr,
                 seqs=np.array(seqs, dtype=object))
        print(f"# {m}: N={len(y)}  complete={int((y==1).sum())} incomplete={int((y==0).sum())} "
              f"face_valid_ratio median={np.median(vr):.2f}  -> {outf}")

if __name__ == "__main__":
    main()
