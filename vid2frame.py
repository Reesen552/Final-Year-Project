import os
import cv2
import glob


def vid2pred():
    Preprocess()



class Preprocess():
    def __init__(self, ratio=.7):
        self.ratio = ratio
        data_folder = 'data'
        labels = sorted(os.listdir(os.path.join(os.getcwd(), '{}/regime_videos'.format(data_folder))))

        video_list = []
        for label in labels:
            video_list.append(
                [mp4 for mp4 in glob.iglob('{}/regime_videos/{}/*.mp4'.format(data_folder, label), recursive=True)])

        #(video -> image)
        if (2 > 1):
            # if not os.path.exists('{}/train'.format(data_folder)):
            for label in labels:
                os.makedirs(os.path.join(os.getcwd(), '{}/images'.format(data_folder), label), exist_ok=True)

            for videos in video_list:
                for i, video in enumerate(videos):
                    self.video2frame(video, '{}/images'.format(data_folder))


    def video2frame(self, video, frame_save_path, count=0):

        folder_name, video_name = video.split('/')[-2], video.split('/')[-1]

        capture = cv2.VideoCapture(video)
        get_frame_rate = round(capture.get(cv2.CAP_PROP_FRAME_COUNT) / 800)

        _, frame = capture.read()

        while True:
            ret, image = capture.read()
            if not ret:
                break

            if (int(capture.get(1)) % get_frame_rate == 0):
                count += 1
                fname = '/{0}_{1:05d}.jpg'.format(video_name, count)
                cv2.imwrite('{}/{}'.format(frame_save_path, fname), image) #NOT SAVING IMAGES

        print("{} images are extracted in {}.".format(count, frame_save_path))


def main():
    vid2pred()


if __name__ == '__main__':
    main()