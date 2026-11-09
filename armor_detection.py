import cv2
import numpy as np

class LightBar:
    def __init__(self, rect):
        self.rect = rect  # minAreaRect
        self.center = rect[0]
        self.size = rect[1]
        self.angle = rect[2]
        # 归一角度到 0~180
        if self.size[0] < self.size[1]:
            self.w, self.h = self.size[1], self.size[0]
            self.angle = (self.angle + 90) % 180
        else:
            self.w, self.h = self.size[0], self.size[1]
        self.aspect_ratio = self.w / (self.h + 1e-6)

def detect_light_bars(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 红色阈值，可自行调，蓝色同理
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bars = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        rect = cv2.minAreaRect(cnt)
        bar = LightBar(rect)
        # 细长灯条筛选
        if bar.aspect_ratio > 2.5:
            bars.append(bar)
    return bars, mask

def pair_light_bars(bars):
    pairs = []
    n = len(bars)
    used = set()
    for i in range(n):
        if i in used:
            continue
        b1 = bars[i]
        best_j = None
        best_score = None
        for j in range(i + 1, n):
            if j in used:
                continue
            b2 = bars[j]
            # 角度接近平行
            angle_diff = abs(b1.angle - b2.angle)
            if angle_diff > 15 and abs(angle_diff - 180) > 15:
                continue
            # 高度接近
            h_diff = abs(b1.h - b2.h)
            if h_diff > max(b1.h, b2.h) * 0.4:
                continue
            # 中点距离
            cx1, cy1 = b1.center
            cx2, cy2 = b2.center
            dist = np.hypot(cx2 - cx1, cy2 - cy1)
            if dist < 10 or dist > 400:
                continue
            score = dist
            if best_score is None or score < best_score:
                best_score = score
                best_j = j
        if best_j is not None:
            pairs.append((b1, bars[best_j]))
            used.add(i)
            used.add(best_j)
    return pairs

def extract_armor_roi(img, pair):
    b1, b2 = pair
    c1 = np.array(b1.center)
    c2 = np.array(b2.center)
    armor_center = (c1 + c2) / 2
    
    pts1 = cv2.boxPoints(b1.rect)
    pts2 = cv2.boxPoints(b2.rect)
    all_pts = np.concatenate([pts1, pts2])
    x_min = int(np.min(all_pts[:, 0]))
    x_max = int(np.max(all_pts[:, 0]))
    y_min = int(np.min(all_pts[:, 1]))
    y_max = int(np.max(all_pts[:, 1]))
    
    h_img, w_img = img.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w_img, x_max)
    y_max = min(h_img, y_max)
    roi = img[y_min:y_max, x_min:x_max]
    return roi, (x_min, y_min, x_max, y_max), armor_center

def visualize(img, bars, pairs):
    vis = img.copy()
 
    for b in bars:
        box = cv2.boxPoints(b.rect)
        box = np.int64(box)
        cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)
    
    for p in pairs:
        _, (x1, y1, x2, y2), _ = extract_armor_roi(img, p)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
    return vis

def main(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("图片读取失败")
        return
    bars, mask = detect_light_bars(img)
    pairs = pair_light_bars(bars)
    print(f"检测灯条数量：{len(bars)}")
    print(f"成功配对装甲板数量：{len(pairs)}")

    for idx, p in enumerate(pairs):
        roi, rect, center = extract_armor_roi(img, p)
        cv2.imwrite(f"armor_roi_{idx}.png", roi)
        print(f"装甲板 {idx} ROI 已保存，中心：{center}")

    vis_img = visualize(img, bars, pairs)
    cv2.imwrite("result_vis.png", vis_img)
    cv2.imwrite("mask.png", mask)
    cv2.imshow("visual", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    main("armor.jpg")
