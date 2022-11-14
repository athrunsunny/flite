import argparse
import os
import sys
from pathlib import Path
import onnx
import onnxruntime
import cv2
import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.dataloader1 import IMG_FORMATS, VID_FORMATS, LoadImages1, letterbox
from utils.general import (LOGGER, check_file, form_net, increment_path, print_args, load_model, Profile,
                           select_device, smart_inference_mode)

from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'faceDetector.onnx', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'img', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=320, help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect-lmk', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    opt = parser.parse_args()
    # opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def scale_coords_landmarks(img1_shape, coords, img0_shape, scale, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords = coords * scale

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


def scale_boxes(img1_shape, boxes, img0_shape, scale, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes = boxes * scale

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


@smart_inference_mode()
def run(
        weights=ROOT / 'faceDetector.onnx',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=320,  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        top_k=5000,
        nms_threshold=0.4,
        keep_top_k=750,
        vis_thres=0.6
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    w = str(weights)
    # Load model
    device = select_device(device)
    model, cfg = form_net('slim', device, phase='inference')

    cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(w, providers=providers)
    output_names = [x.name for x in session.get_outputs()]

    # Dataloader
    bs = 1  # batch_size
    dataset = LoadImages1(source, img_size=imgsz, vid_stride=vid_stride, oresize=False, auto=False)

    # Run inference
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s, resize in dataset:
        p = Path(path)  # to Path
        save_path = str(save_dir / p.name)  # im.jpg

        im_height, im_width = im.shape[1:]
        scale = torch.Tensor([im_width, im_height, im_width, im_height])
        scale = scale.to(device)

        with dt[0]:
            im = torch.from_numpy(im).to(device)
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        im = im.cpu().numpy()
        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            loc, conf, landms = session.run(output_names, {session.get_inputs()[0].name: im})
            loc = torch.from_numpy(loc).to(device)
            conf = torch.from_numpy(conf).to(device)
            landms = torch.from_numpy(landms).to(device)
        # NMS
        with dt[2]:
            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data

            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = scale_boxes(im.shape[2:], boxes, im0s.shape, scale)
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
            scale1 = torch.Tensor([im.shape[3], im.shape[2], im.shape[3], im.shape[2],
                                   im.shape[3], im.shape[2], im.shape[3], im.shape[2],
                                   im.shape[3], im.shape[2]])
            scale1 = scale1.to(device)
            landms = scale_coords_landmarks(im.shape[2:], landms, im0s.shape, scale1)
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > conf_thres)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, nms_threshold)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:keep_top_k, :]
            landms = landms[:keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)

            # show image
            if save_img:
                for b in dets:
                    if b[4] < vis_thres:
                        continue
                    text = "{:.4f}".format(b[4])
                    b = list(map(int, b))
                    cv2.rectangle(im0s, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                    cx = b[0]
                    cy = b[1] + 12
                    cv2.putText(im0s, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                    # landms
                    cv2.circle(im0s, (b[5], b[6]), 1, (0, 0, 255), 4)
                    cv2.circle(im0s, (b[7], b[8]), 1, (0, 255, 255), 4)
                    cv2.circle(im0s, (b[9], b[10]), 1, (255, 0, 255), 4)
                    cv2.circle(im0s, (b[11], b[12]), 1, (0, 255, 0), 4)
                    cv2.circle(im0s, (b[13], b[14]), 1, (255, 0, 0), 4)

                cv2.imwrite(save_path, im0s)
        LOGGER.info(f"{s}{'' if len(dets) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
