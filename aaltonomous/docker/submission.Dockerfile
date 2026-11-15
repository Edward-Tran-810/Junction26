# Team submission: top-down three-finger grasping controller (classical stack: Perception v2 + language module +
# near-contact top-down grasp, aaltonomous/baselines/classical, grasp.mode=topdown). Spec (SUBMISSION.md): x86_64,
# CUDA >= 12.8 base, policy server on port 8000 within 120 s, weights inside the image (this policy has NO learned
# weights: geometric segmenter, rule-based language parser, IK planner), fits in 32 GB VRAM (it uses no GPU at all).
#
# Build from the repo root:  docker build -f aaltonomous/docker/submission.Dockerfile -t <image>:<tag> .
# Test locally:              docker run --rm -p 8000:8000 <image>:<tag>       (no GPU needed; --gpus all also works)
#                            python -m aaltonomous.deploy.replay --synthetic 100 --policy ws://localhost:8000
# Dependencies and file list follow the reference classical submission (baseline.Dockerfile).
FROM nvidia/cuda:12.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip libgomp1 && rm -rf /var/lib/apt/lists/*
# Pinocchio wheel (pin 3.3.1) needs numpy >= 2.2 and libgomp1 (SUBMISSION.md section 7)
RUN pip3 install --no-cache-dir numpy==2.2.6 scipy==1.15.2 opencv-python-headless==4.10.0.84 pyyaml \
        msgpack "websockets>=12" pin==3.3.1

WORKDIR /app
COPY third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf third_party/xr_teleoperate/assets/g1/
COPY aaltonomous/__init__.py aaltonomous/
COPY aaltonomous/deploy/__init__.py aaltonomous/deploy/protocol.py aaltonomous/deploy/policy_server.py aaltonomous/deploy/
COPY aaltonomous/interface/__init__.py aaltonomous/interface/safety.py aaltonomous/interface/calibration.py \
     aaltonomous/interface/tasks.py \
     aaltonomous/interface/joint_spec.yaml aaltonomous/interface/calibration.yaml aaltonomous/interface/
COPY aaltonomous/scene/__init__.py aaltonomous/scene/scene.py aaltonomous/scene/scene.yaml aaltonomous/scene/
COPY aaltonomous/baselines/__init__.py aaltonomous/baselines/
COPY aaltonomous/baselines/classical/*.py aaltonomous/baselines/classical/config.yaml aaltonomous/baselines/classical/
# fail the BUILD (not the finals start) if the policy cannot be constructed; and record the active grasp mode
RUN python3 -c "from aaltonomous.baselines.classical.policy_server import ClassicalServerPolicy as P; p = P(); \
from aaltonomous.baselines.classical.config import load_config as L; c = L(); assert c['grasp']['mode'] == 'topdown', c['grasp']['mode']; \
print('grasp.mode', c['grasp']['mode'], 'profile', c['profile'])"

EXPOSE 8000
CMD ["python3", "-m", "aaltonomous.baselines.classical.policy_server"]
