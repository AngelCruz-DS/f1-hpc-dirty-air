import os, glob, cv2

os.makedirs("data/frames/preview", exist_ok=True)
imgs = sorted(glob.glob("data/frames/val/images/*.jpg"))[:15]

for p in imgs:
    img = cv2.imread(p)
    h, w, _ = img.shape
    lbl = p.replace("images", "labels").replace(".jpg", ".txt")
    if os.path.exists(lbl):
        with open(lbl) as f:
            for line in f:
                _, xc, yc, bw, bh = map(float, line.split())
                x1, y1 = int((xc - bw/2)*w), int((yc - bh/2)*h)
                x2, y2 = int((xc + bw/2)*w), int((yc + bh/2)*h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(os.path.join("data/frames/preview", os.path.basename(p)), img)

print("Vista previa guardada en data/frames/preview (15 muestras).")
