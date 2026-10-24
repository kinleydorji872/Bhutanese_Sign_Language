"""Optional local webcam preview. This is not the upload-based web app."""
import cv2

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check camera permissions.")
    print("Press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Bhutanese Sign Language - camera preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
