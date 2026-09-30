"""
가위바위보 대전 게임 (HandDetector + MobileNetV2, LiteRT)
- EX_07_Board_RPS_MobileNetV2_HD_Augmentation.py 기반, none 클래스 추가
- 화면 가운데 선 기준 왼쪽 손 = Player 1, 오른쪽 손 = Player 2
- 모드: 2인 대전(기본) / 묵찌빠 / PC 대전(컴퓨터와 가위바위보)
- 키: S 시작 / A 자동판정 on/off / M 모드 변경 / R 점수 초기화 / D 손가락 상태 표시 / Q 종료
"""
import os
import time
import random
from collections import Counter

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from cvzone.HandTrackingModule import HandDetector

# ------------------------------------------------------------
# 설정값
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# none 포함 모델(새로 학습)이 있으면 그것을, 없으면 교수님 모델을 사용
NEW_MODEL = os.path.join(BASE_DIR, 'RPS_MobileNetV2_None.tflite')
OLD_MODEL = os.path.join(BASE_DIR, 'RPS_MobileNetV2_Augmentation.tflite')
MODEL_PATH = NEW_MODEL if os.path.exists(NEW_MODEL) else OLD_MODEL

CLASSES = ['scissors', 'rock', 'paper', 'none']   # 학습 때 클래스 순서와 동일해야 함
NONE_ID = 3
IMG_SIZE = 224
CAM_W, CAM_H = 640, 480     # 느리면 320, 240 으로 낮추기
OFFSET = int(30 * CAM_W / 320)   # 예제(320x240)의 offset 30을 해상도에 맞춰 조정
CONF_THR = 0.5              # 확률이 이보다 낮으면 판정에 쓰지 않음

HOLD_SEC = 1.0        # 자동판정: 손 모양을 이 시간 동안 유지하면 판정
COUNT_STEP_SEC = 0.6  # 카운트다운 한 박자 길이
CAPTURE_SEC = 0.4     # 카운트다운 직후 손 모양을 모으는(투표) 시간
RESULT_SEC = 2.5      # 결과 표시 시간

SHOW_FINGERS = False
VERSION = 'v5 (muk-jji-ppa auto next)'

MODES = ['RPS', 'MJB', 'PC']
MODE_NAME = {'RPS': '2P BATTLE', 'MJB': 'MUK-JJI-PPA', 'PC': 'VS PC'}

P1_COLOR = (255, 128, 0)   # BGR
P2_COLOR = (0, 128, 255)
WHITE, YELLOW, GREEN = (255, 255, 255), (0, 255, 255), (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ------------------------------------------------------------
# 모델 (교수님 예제와 같은 방식으로 입력 준비)
# ------------------------------------------------------------
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_dtype = input_details[0]['dtype']
NUM_CLASSES = int(output_details[0]['shape'][-1])   # 3이면 none 없는 모델, 4면 none 포함
print('RPS game', VERSION)
print('model:', os.path.basename(MODEL_PATH))
print('model input shape:', input_details[0]['shape'], '/ classes:', CLASSES[:NUM_CLASSES])

hd = HandDetector(maxHands=2)


def make_square_img(img):
    """예제와 동일: 흰 배경 정사각형에 손 이미지를 비율 유지해서 넣기"""
    ho, wo = img.shape[0], img.shape[1]
    wbg = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
    if ho / wo > 1:   # portrait
        wk = int(wo * IMG_SIZE / ho)
        img = cv2.resize(img, (wk, IMG_SIZE))
        d = (IMG_SIZE - img.shape[1]) // 2
        wbg[:, d:img.shape[1] + d] = img
    else:             # landscape
        hk = int(ho * IMG_SIZE / wo)
        img = cv2.resize(img, (IMG_SIZE, hk))
        d = (IMG_SIZE - img.shape[0]) // 2
        wbg[d:img.shape[0] + d, :] = img
    return wbg


def to_prob(out):
    """출력이 확률이 아니면(softmax 전 값이면) softmax 적용"""
    out = out.astype(np.float32)
    if out.min() < 0 or abs(out.sum() - 1.0) > 0.05:
        e = np.exp(out - out.max())
        out = e / e.sum()
    return out


# 손가락 관절 번호 (MediaPipe 손 랜드마크): 0=손목, (PIP 관절, 손끝)
FINGERS = [(6, 8), (10, 12), (14, 16), (18, 20)]   # 검지, 중지, 약지, 새끼


def fingers_extended(lm):
    """손끝이 가운데 관절보다 손목에서 더 멀면 '펴진 손가락' (손 방향과 무관)"""
    w = np.array(lm[0][:2], dtype=float)
    d = lambda i: np.linalg.norm(np.array(lm[i][:2], dtype=float) - w)
    return [d(tip) > d(pip) for pip, tip in FINGERS]


THUMB_RATIO = 1.1   # 엄지 끝~새끼 뿌리 거리가 손바닥 길이의 이 배수보다 크면 '엄지 펴짐'


def thumb_extended(lm):
    """엄지는 옆으로 펴지니까, 엄지 끝(4)이 새끼 뿌리(17)에서 멀리 떨어졌는지로 판단"""
    p = lambda i: np.array(lm[i][:2], dtype=float)
    palm = np.linalg.norm(p(9) - p(0))            # 손목~중지 뿌리 = 손바닥 길이
    return bool(np.linalg.norm(p(4) - p(17)) > THUMB_RATIO * palm)


def finger_state(lm):
    """[엄지, 검지, 중지, 약지, 새끼] 펴짐 여부"""
    return [thumb_extended(lm)] + fingers_extended(lm)


# 클래스별 올바른 손가락 모양 [엄지, 검지, 중지, 약지, 새끼] (1=펴짐, 0=접힘, None=상관없음)
SHAPES = {
    0: [0, 1, 1, 0, 0],       # 가위: 검지, 중지만
    1: [None, 0, 0, 0, 0],    # 바위: 네 손가락 모두 접힘 (엄지는 상관없음)
    2: [1, 1, 1, 1, 1],       # 보: 다섯 손가락 모두 펴짐
}


def shape_ok(cls, lm):
    """모델이 고른 클래스와 실제 손가락 모양이 맞는지 확인"""
    expected = SHAPES.get(cls)
    if expected is None:
        return True
    return all(e is None or bool(e) == f for e, f in zip(expected, finger_state(lm)))


def classify(frame, bbox, lm=None):
    """손 bbox -> (잘라낸 영역 좌표, 클래스, 확률)"""
    H, W = frame.shape[:2]
    x, y, w, h = bbox
    # 예제와 같은 방식으로 여유(offset) 추가: 좌/상/우만, 화면 밖은 잘라냄
    x1, y1 = max(0, x - OFFSET), max(0, y - OFFSET)
    x2, y2 = min(W, x + w + OFFSET), min(H, y + h)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    img = make_square_img(frame[y1:y2, x1:x2])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)      # 정규화는 모델 안에서 함 (예제와 동일)
    interpreter.set_tensor(input_details[0]['index'], img.astype(input_dtype))
    interpreter.invoke()
    prob = to_prob(interpreter.get_tensor(output_details[0]['index'])[0])
    cls = int(np.argmax(prob))
    conf = float(prob[cls])
    # 이중 체크: 모델 결과와 손가락 모양이 다르면 none (예: 손가락 1, 3, 4)
    if lm is not None and not shape_ok(cls, lm):
        cls = NONE_ID
    fs = ''.join('1' if f else '0' for f in finger_state(lm)) if lm is not None else '-----'
    return {'box': (x1, y1, x2, y2), 'cls': cls, 'conf': conf, 'fingers': fs,
            'cx': x + w / 2, 'area': w * h}


def find_players(frame, mode):
    """손 찾기 -> 분류 -> P1/P2 배정"""
    res = hd.findHands(frame, draw=False)
    hands = res[0] if isinstance(res, tuple) else res   # cvzone 버전 차이 대응
    preds = [p for p in (classify(frame, h['bbox'], h.get('lmList')) for h in (hands or [])) if p]

    biggest = lambda lst: max(lst, key=lambda p: p['area'], default=None)
    if mode == 'PC':                      # PC 대전: 화면에서 가장 큰 손 = P1
        return biggest(preds), None
    mid = frame.shape[1] / 2
    return (biggest([p for p in preds if p['cx'] < mid]),
            biggest([p for p in preds if p['cx'] >= mid]))


def gesture(p):
    """손이 없거나, none이거나, 확신이 낮으면 None -> 판정에 안 씀"""
    if p is None or p['cls'] == NONE_ID or p['conf'] < CONF_THR:
        return None
    return p['cls']


# ------------------------------------------------------------
# 승패 판정: 0=scissors, 1=rock, 2=paper
#   (g1 - g2) % 3 -> 0: 무승부, 1: P1 승, 2: P2 승
# ------------------------------------------------------------
def judge(g1, g2):
    return (g1 - g2) % 3


def majority(votes):
    return Counter(votes).most_common(1)[0][0] if votes else None


# ------------------------------------------------------------
# 게임 FSM: IDLE -> COUNTDOWN -> CAPTURE -> RESULT -> IDLE
# ------------------------------------------------------------
class Game:
    IDLE, COUNTDOWN, CAPTURE, RESULT = range(4)

    def __init__(self):
        self.mode = 'RPS'
        self.auto = False
        self.score = [0, 0]
        self.reset_round()

    def reset_round(self):
        self.state = self.IDLE
        self.attacker = 0          # 묵찌빠 공격권 (0=없음, 1/2)
        self.t0 = 0.0
        self.votes = ([], [])
        self.hold = [None, None]   # 자동판정용 (손모양, 유지 시작 시각)
        self.armed = True
        self.last = None
        self.msg = ''
        self.chain = False         # 묵찌빠: 결과 후 다음 판 자동 시작 여부

    def players(self):             # 판정에 필요한 사람 수
        return 1 if self.mode == 'PC' else 2

    def start(self, now):
        if self.state == self.IDLE:
            self.state, self.t0 = self.COUNTDOWN, now

    def update(self, now, g1, g2):
        gs = (g1, g2)[:self.players()]
        if self.state == self.IDLE:
            if self.auto:
                self._update_hold(now, g1, g2)
                if any(g is None for g in gs):
                    self.armed = True          # 손을 내리면 다음 판 준비
                elif self.armed and self._held(now):
                    self._resolve(now, g1, g2)

        elif self.state == self.COUNTDOWN:
            if now - self.t0 >= 3 * COUNT_STEP_SEC:
                self.state, self.t0, self.votes = self.CAPTURE, now, ([], [])

        elif self.state == self.CAPTURE:
            for i, g in enumerate((g1, g2)):
                if g is not None:
                    self.votes[i].append(g)
            if now - self.t0 >= CAPTURE_SEC:
                v1, v2 = majority(self.votes[0]), majority(self.votes[1])
                if v1 is None or (self.mode != 'PC' and v2 is None):
                    self.last, self.msg = None, 'NO HAND - try again'
                    self.state, self.t0 = self.RESULT, now
                else:
                    self._resolve(now, v1, v2)

        elif self.state == self.RESULT:
            if now - self.t0 >= RESULT_SEC:
                if self.chain and not self.auto:   # 묵찌빠 진행 중이면 다음 판 자동 시작
                    self.state, self.t0 = self.COUNTDOWN, now
                else:
                    self.state = self.IDLE
                self.chain = False

    def _update_hold(self, now, g1, g2):
        for i, g in enumerate((g1, g2)):
            if self.hold[i] is None or self.hold[i][0] != g:
                self.hold[i] = (g, now)

    def _held(self, now):
        hs = self.hold[:self.players()]
        return all(h and h[0] is not None and now - h[1] >= HOLD_SEC for h in hs)

    def hold_progress(self, now):
        hs = self.hold[:self.players()]
        if not self.auto or self.state != self.IDLE or not self.armed:
            return 0.0
        if any(h is None or h[0] is None for h in hs):
            return 0.0
        return min(1.0, min(now - h[1] for h in hs) / HOLD_SEC)

    def _resolve(self, now, g1, g2):
        if self.mode == 'PC':
            g2 = random.randrange(3)           # 컴퓨터는 무작위
        r = judge(g1, g2)
        name2 = 'PC' if self.mode == 'PC' else 'PLAYER 2'
        if self.mode in ('RPS', 'PC'):
            if r == 0:
                self.msg = 'DRAW'
            else:
                self.score[r - 1] += 1
                self.msg = 'PLAYER 1 WIN!' if r == 1 else f'{name2} WIN!'
        else:  # 묵찌빠
            if self.attacker == 0:             # 공격권 정하기
                if r == 0:
                    self.msg = 'DRAW - again'
                else:
                    self.attacker, self.msg = r, f'P{r} ATTACK!'
                self.chain = True                # 비겨도, 공격권이 생겨도 다음 판으로
            elif g1 == g2:                     # 같은 손 -> 공격자 승리
                w = self.attacker
                self.score[w - 1] += 1
                self.msg, self.attacker = f'PLAYER {w} WIN!', 0
            else:                              # 다른 손 -> 이긴 사람이 새 공격자
                self.attacker, self.msg = r, f'P{r} ATTACK!'
                self.chain = True
        self.last = (g1, g2)
        self.state, self.t0 = self.RESULT, now
        self.armed = False
        self.hold = [None, None]

    def countdown_word(self, now):
        words = ['MUK', 'JJI', 'PPA!'] if (self.mode == 'MJB' and self.attacker) \
            else ['GAWI', 'BAWI', 'BO!']
        return words[min(2, int((now - self.t0) / COUNT_STEP_SEC))]


# ------------------------------------------------------------
# 화면 그리기 (글자 크기는 화면 폭에 맞춰 자동 조절)
# ------------------------------------------------------------
def put(frame, text, org, scale, color, thick=2):
    k = frame.shape[1] / 640
    s, t = scale * k, max(1, int(round(thick * k)))
    cv2.putText(frame, text, org, FONT, s, (0, 0, 0), t + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, s, color, t, cv2.LINE_AA)


def put_center(frame, text, y, scale, color, thick=2):
    k = frame.shape[1] / 640
    (tw, _), _ = cv2.getTextSize(text, FONT, scale * k, max(1, int(round(thick * k))))
    put(frame, text, ((frame.shape[1] - tw) // 2, y), scale, color, thick)


def draw(frame, p1, p2, game, now, fps):
    h, w = frame.shape[:2]
    k = w / 640
    if game.mode != 'PC':
        cv2.line(frame, (w // 2, 0), (w // 2, h), WHITE, 1)

    for p, color, name in ((p1, P1_COLOR, 'P1'), (p2, P2_COLOR, 'P2')):
        if p is None:
            continue
        x1, y1, x2, y2 = p['box']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        put(frame, f"{name} {CLASSES[p['cls']]} {p['conf']:.2f}",
            (x1, max(int(20 * k), y1 - 8)), 0.6, color)
        if SHOW_FINGERS:   # D 키: 손가락 상태(엄지 검지 중지 약지 새끼, 1=펴짐) 표시
            put(frame, p['fingers'], (x1, min(h - 5, y2 + int(22 * k))), 0.6, color)

    name2 = 'PC' if game.mode == 'PC' else 'P2'
    put(frame, f"{MODE_NAME[game.mode]} | AUTO {'ON' if game.auto else 'OFF'} | FPS {fps:.1f}",
        (int(10 * k), int(25 * k)), 0.55, YELLOW, 1)
    put(frame, f'P1 {game.score[0]}', (int(10 * k), int(60 * k)), 0.9, P1_COLOR)
    put(frame, f'{name2} {game.score[1]}', (w - int(120 * k), int(60 * k)), 0.9, P2_COLOR)
    if game.mode == 'MJB' and game.attacker:
        put_center(frame, f'ATTACK: P{game.attacker}', int(60 * k), 0.7,
                   P1_COLOR if game.attacker == 1 else P2_COLOR)

    if game.state == Game.COUNTDOWN:
        put_center(frame, game.countdown_word(now), h // 2, 2.0, YELLOW, 4)
    elif game.state == Game.CAPTURE:
        put_center(frame, 'CHECK...', h // 2, 1.5, YELLOW, 3)
    elif game.state == Game.RESULT:
        put_center(frame, game.msg, h // 2, 1.4, GREEN, 3)
        if game.last:
            put_center(frame, f'{CLASSES[game.last[0]]}  vs  {CLASSES[game.last[1]]}',
                       h // 2 + int(45 * k), 0.8, WHITE)
        if game.chain and not game.auto:
            put_center(frame, 'next round...', h // 2 + int(85 * k), 0.7, YELLOW)
    else:
        prog = game.hold_progress(now)
        if prog > 0:
            x0, x1_ = int(100 * k), w - int(100 * k)
            y0 = h - int(70 * k)
            cv2.rectangle(frame, (x0, y0), (x0 + int((x1_ - x0) * prog), y0 + int(15 * k)), GREEN, -1)
            cv2.rectangle(frame, (x0, y0), (x1_, y0 + int(15 * k)), WHITE, 1)
        elif not game.auto:
            put_center(frame, 'Press S', h // 2, 1.0, WHITE)

    put(frame, 'S:start A:auto M:mode R:reset D:fingers Q:quit', (int(10 * k), h - int(15 * k)), 0.5, WHITE, 1)


# ------------------------------------------------------------
# 메인 루프
# ------------------------------------------------------------
def main():
    global SHOW_FINGERS
    game = Game()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow('RPS Battle', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('RPS Battle', 640 + 40, 480 + 60)
    prev = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)   # 거울 모드: 왼쪽에 선 사람이 화면 왼쪽(P1)

        p1, p2 = find_players(frame, game.mode)
        now = time.time()
        game.update(now, gesture(p1), gesture(p2))

        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        draw(frame, p1, p2, game, now, fps)
        cv2.imshow('RPS Battle', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            game.start(now)
        elif key == ord('a'):
            game.auto = not game.auto
            game.reset_round()
        elif key == ord('m'):
            game.mode = MODES[(MODES.index(game.mode) + 1) % len(MODES)]
            game.score = [0, 0]
            game.reset_round()
        elif key == ord('d'):
            SHOW_FINGERS = not SHOW_FINGERS
        elif key == ord('r'):
            game.score = [0, 0]
            game.reset_round()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
