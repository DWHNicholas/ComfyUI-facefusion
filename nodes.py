import mimetypes
import urllib.request

from PIL import Image

from facefusion.core import common_pre_check, conditional_append_reference_faces
from facefusion.face_analyser import get_many_faces, get_one_face
from facefusion.face_store import append_reference_face, clear_reference_faces
from facefusion.processors.choices import face_debugger_items

opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent',
                      'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36'),
                     ('Connection', 'keep-alive')]
urllib.request.install_opener(opener)

from .utils import batch_tensor_to_pil, batched_pil_to_tensor, tensor_to_pil
import tempfile, requests, uuid, os

from facefusion.core import conditional_process
try:
    import torch
    import folder_paths
except:
    folder_paths = None

from facefusion.choices import face_mask_types,face_selector_orders,face_selector_modes,\
    face_mask_regions as total_face_mask_regions,face_detector_set

# =================================
def get_mime_type(file_path):
    # 获取文件的 MIME 类型
    mime_type, _ = mimetypes.guess_type(file_path)

    # 如果无法猜测类型，返回默认类型
    if mime_type is None:
        return 'application/octet-stream'

    return mime_type


def empty_torch():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass


def debug(time):

    try:
        from facefusion.inference_manager import INFERENCE_POOLS
        inf=INFERENCE_POOLS['cli']['facefusion.face_detector.yoloface.cuda']["yoloface"]
        onnxruntime_provide=inf._providers
    except:
        onnxruntime_provide="cpu"

    return f"info:[onnx:{onnxruntime_provide}]\n[download_time:{time}]"

common_pre_check()


common_input_dict={
    "single_source_image": ("IMAGE",),  # Single source image
    "device": (["cpu", "cuda"], {"default": "cuda"}),  # Execution provider
    "face_detector_score": ("FLOAT", {"default": 0.65, "min": 0, "max": 1, "step": 0.02}),
    # Face detector score
    "mask_blur": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.05}),  # Face mask blur
    "landmarker_score": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.05}),
    # Face landmarker score
    "face_enhance_blend": ("FLOAT", {"default": 30, "min": 0, "max": 100, "step": 1}),
    "face_selector_order": (face_selector_orders,{"default": face_selector_orders[0]}),
    "face_selector_mode": (face_selector_modes, {"default": face_selector_modes[0]}),
    "reference_face_position": ("INT", {"default": 0}),
    "reference_face_distance": ("FLOAT", {"max": 2.0, "min": 0.0, "default": 0.6}),
}
common_input_dict2 = {
    "reference_face_image": ("IMAGE", ),
    "face_detector_model": (list(face_detector_set.keys()), {"default": list(face_detector_set.keys())[-1]}),
    "faceswap_poisson_blend": ("FLOAT", {"default": 1., "min": 0, "max": 1., "step": 0.05}),
    "face_mask_types": (face_mask_types, {"default": face_mask_types[0]}),
    **{i:("BOOLEAN", {"default": True}) for i in total_face_mask_regions},
    "face_debug": ("BOOLEAN", {"default": False}),
    **{i:("BOOLEAN", {"default": i in ['face-landmark-5/68', 'face-mask']}) for i in face_debugger_items},

}




def facefusion_run(source_path, target_path: str, output_path, provider, face_selector_mode, reference_face_position,
                   reference_face_distance, working=conditional_process,detector_score=0.6, mask_blur=0.3,faceswap_poisson_blend=1.,
                   face_enhance_blend=0., landmarker_score=0.5, thread_count=1, face_selector_order=None,face_detector_model='yoloface',
                   reference_face_image=None,face_mask_types='box',face_mask_regions=tuple(total_face_mask_regions),
                   debug=False,debug_items=('face-landmark-5/68', 'face-mask')):
    from facefusion.vision import detect_image_resolution, pack_resolution, detect_video_resolution, detect_video_fps
    from facefusion.filesystem import is_video, is_image
    from facefusion import state_manager
    the_processors = ['face_swapper', ]
    if face_enhance_blend > 0.:
        the_processors.append('face_enhancer')
    if debug > 0.:
        the_processors.append('face_debugger')
    apply_state_item = state_manager.set_item
    apply_state_item('processors', the_processors)
    apply_state_item('face_detector_angles', [0])
    apply_state_item('face_selector_order', face_selector_order, )
    apply_state_item('face_debugger_items', debug_items, )


    # ===
    apply_state_item('faceswap_poisson_blend', faceswap_poisson_blend)
    apply_state_item('face_selector_mode', face_selector_mode, )
    apply_state_item('reference_face_position', reference_face_position, )
    apply_state_item('reference_face_distance', reference_face_distance, )
    apply_state_item('skip_download', False, )
    apply_state_item('execution_thread_count', thread_count, )
    apply_state_item('face_enhancer_blend', face_enhance_blend)
    apply_state_item('source_paths', source_path)
    apply_state_item('target_path', target_path)
    apply_state_item('output_path', output_path)
    apply_state_item('execution_providers', provider)
    apply_state_item('face_detector_score', detector_score)
    apply_state_item('face_mask_blur', mask_blur)
    apply_state_item('face_landmarker_score', landmarker_score)
    apply_state_item('face_detector_model', face_detector_model, )
    apply_state_item('face_detector_size', '640x640', )
    apply_state_item('face_landmarker_model', '2dfan4', )
    apply_state_item('reference_frame_number', 0, )
    apply_state_item('face_mask_types', [face_mask_types], )
    apply_state_item('face_mask_padding', (0, 0, 0, 0), )
    apply_state_item('temp_frame_format', 'png', )
    apply_state_item('output_image_quality', 80, )
    apply_state_item('output_audio_encoder', 'aac', )
    apply_state_item('output_video_encoder', 'libx264', )
    apply_state_item('output_video_preset', 'veryfast', )
    apply_state_item('output_video_quality', 80, )
    apply_state_item('age_modifier_model', 'styleganex_age', )
    apply_state_item('age_modifier_direction', 0, )
    apply_state_item('expression_restorer_model', 'live_portrait', )
    apply_state_item('expression_restorer_factor', 80, )
    apply_state_item('face_editor_model', 'live_portrait', )
    apply_state_item('face_enhancer_model', 'gfpgan_1.4', )
    apply_state_item('face_swapper_model', 'inswapper_128_fp16', )
    apply_state_item('face_swapper_pixel_boost', '128x128', )
    apply_state_item('frame_enhancer_model', 'span_kendata_x4', )
    apply_state_item('frame_enhancer_blend', 80, )
    apply_state_item('open_browser', False, )
    apply_state_item('execution_queue_count', 1, )
    apply_state_item('video_memory_strategy', None, )
    apply_state_item('face_mask_regions',face_mask_regions if face_mask_regions else total_face_mask_regions)
    # apply_state_item('execution_device_id', '0')
    if is_image(image_path=target_path):
        image_resolution = detect_image_resolution(target_path)
        apply_state_item('output_image_resolution', pack_resolution(image_resolution))
    elif is_video(video_path=target_path):
        video_resolution = detect_video_resolution(target_path)
        apply_state_item('output_video_resolution', pack_resolution(video_resolution))
        apply_state_item('output_video_fps', int(detect_video_fps(target_path)))
    from facefusion.core import processors_pre_check
    import numpy as np
    res=None
    if processors_pre_check():
        if reference_face_image is not None:
            pil_img:Image.Image = tensor_to_pil(img_tensor=reference_face_image).convert("RGB")
            from facefusion.face_selector import sort_and_filter_faces
            reference_frame=np.uint8(pil_img)[..., ::-1]
            reference_faces = sort_and_filter_faces(get_many_faces([reference_frame]))
            reference_face = get_one_face(reference_faces)
            append_reference_face('reference', reference_face)
        res=working()
        clear_reference_faces()
    empty_torch()
    if isinstance(res,torch.Tensor):
        return res
    return output_path


class WD_FaceFusion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                **common_input_dict
            },
            "optional": {
                **common_input_dict2
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "WDTRIP"

    def execute(self, image, single_source_image, device, face_detector_score, mask_blur, landmarker_score,faceswap_poisson_blend,
                face_enhance_blend,face_selector_order,face_selector_mode,reference_face_position,reference_face_distance,
                face_mask_types='box',reference_face_image=None,**kwargs):
        source_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        tensor_to_pil(single_source_image).save(source_path)
        source_paths = [source_path]

        target_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        batch_tensor_to_pil(image)[0].save(target_path)

        output_dir = folder_paths.get_output_directory()
        full_output_folder, _, _, _, _, = folder_paths.get_save_image_path("WD_", output_dir)
        file = f"{uuid.uuid4()}.{target_path.split('.')[-1]}"
        output_path = os.path.join(full_output_folder, file)
        facefusion_run(source_paths,
            target_path,
            output_path,
            provider=[device],
            detector_score=face_detector_score,
            mask_blur=mask_blur,
            face_enhance_blend=face_enhance_blend,
            landmarker_score=landmarker_score,
            thread_count=1,
            face_selector_order=face_selector_order,
            face_selector_mode=face_selector_mode,
            reference_face_position=reference_face_position,
            reference_face_distance=reference_face_distance,
            face_mask_types=face_mask_types,
            faceswap_poisson_blend=faceswap_poisson_blend,
            face_mask_regions=[k for k in total_face_mask_regions if kwargs.get(k)],
            face_detector_model=kwargs.get('face_detector_model','yoloface'),
            debug=kwargs.get("face_debug",False),
            debug_items=[k for k in face_debugger_items if kwargs.get(k)],
            reference_face_image=reference_face_image
            )
        result = batched_pil_to_tensor([Image.open(output_path)])
        return (result,)


class WD_FaceFusion_Video:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "thread_count": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
                **common_input_dict
            },
            "optional": {
                "video": ("PATH",),
                "video_url": ("STRING", {
                    "default": "https://exsample.mp4",
                    "defaultBehavior": "input"
                }),
                **common_input_dict2
            }
        }

    RETURN_TYPES = ("PATH","STRING")
    RETURN_NAMES = ("scenes_video","DEBUG_STR")
    FUNCTION = "execute"
    CATEGORY = "WDTRIP"

    def execute(self, video_url, single_source_image, device, face_detector_score, mask_blur, landmarker_score,faceswap_poisson_blend,
                face_enhance_blend, thread_count, face_selector_order, face_selector_mode, reference_face_position,
                face_mask_types='box',reference_face_distance=0.6, video=None,reference_face_image=None,**kwargs):
        # Download the video to a temporary file
        if video is None and (video_url is None or video_url.strip() == ""):
            raise ValueError("Either video_url or video path must be provided")
        if video is not None:
            target_path = video
            time_sec=0
        else:
            import time
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                start=time.time()
                response = requests.get(video_url.strip(), stream=True)
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                end = time.time()
            time_sec=int(end-start)
            target_path = temp_file.name
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, _, subfolder, _, = folder_paths.get_save_image_path("WD_", output_dir)
        file = f"{uuid.uuid4()}.{target_path.split('.')[-1]}"
        output_path = os.path.join(full_output_folder, file)
        source_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        tensor_to_pil(single_source_image).save(source_path)
        source_paths = [source_path]
        facefusion_run(source_paths,
            target_path,
            output_path,
            provider=[device],
            detector_score=face_detector_score,
            mask_blur=mask_blur,
            face_enhance_blend=face_enhance_blend,
            landmarker_score=landmarker_score,
            thread_count=thread_count,
            face_selector_order=face_selector_order,
            face_selector_mode=face_selector_mode,
            reference_face_position=reference_face_position,
            reference_face_distance=reference_face_distance,
            face_mask_types=face_mask_types,
            faceswap_poisson_blend=faceswap_poisson_blend,
            face_mask_regions=[k for k in total_face_mask_regions if kwargs.get(k)],
            face_detector_model=kwargs.get('face_detector_model', 'yoloface'),
            debug=kwargs.get("face_debug", False),
            debug_items=[k for k in face_debugger_items if kwargs.get(k)],
            reference_face_image=reference_face_image
                       )
        return {"ui":{"video":[file,output_path]}, "result": (output_path,debug(time_sec))}


class WD_FaceFusion_Video2:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "thread_count": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
                **common_input_dict
            },
            "optional": {
                "video": ("PATH",),
                "video_url": ("STRING", {
                    "default": "https://exsample.mp4",
                    "defaultBehavior": "input"
                }),
                **common_input_dict2
            }
        }

    RETURN_TYPES = ("IMAGE","FLOAT","STRING")
    RETURN_NAMES = ("images","fps","debug_str")
    FUNCTION = "execute"
    CATEGORY = "WDTRIP"

    def execute(self, video_url, single_source_image, device, face_detector_score, mask_blur, landmarker_score,faceswap_poisson_blend,
                face_enhance_blend, thread_count, face_selector_order, face_selector_mode, reference_face_position,
                face_mask_types='box',reference_face_distance=0.6, video=None,reference_face_image=None,**kwargs):
        # Download the video to a temporary file
        if video is None and (video_url is None or video_url.strip() == ""):
            raise ValueError("Either video_url or video path must be provided")
        if video is not None:
            target_path = video
            time_sec=0
        else:
            import time
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                start=time.time()
                response = requests.get(video_url.strip(), stream=True)
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                end = time.time()
            time_sec=int(end-start)
            target_path = temp_file.name
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, _, subfolder, _, = folder_paths.get_save_image_path("WD_", output_dir)
        file = f"{uuid.uuid4()}.{target_path.split('.')[-1]}"
        output_path = os.path.join(full_output_folder, file)
        source_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        tensor_to_pil(single_source_image).save(source_path)
        source_paths = [source_path]
        from facefusion.vision import detect_video_fps
        fps=detect_video_fps(target_path)
        images=facefusion_run(source_paths,
            target_path,
            output_path,
            working=self.process_video2,
            provider=[device],
            detector_score=face_detector_score,
            mask_blur=mask_blur,
            face_enhance_blend=face_enhance_blend,
            landmarker_score=landmarker_score,
            thread_count=thread_count,
            face_selector_order=face_selector_order,
            face_selector_mode=face_selector_mode,
            reference_face_position=reference_face_position,
            reference_face_distance=reference_face_distance,
            face_mask_types=face_mask_types,
            faceswap_poisson_blend=faceswap_poisson_blend,
            face_mask_regions=[k for k in total_face_mask_regions if kwargs.get(k)],
            face_detector_model=kwargs.get('face_detector_model', 'yoloface'),
            debug=kwargs.get("face_debug", False),
            debug_items=[k for k in face_debugger_items if kwargs.get(k)],
            reference_face_image=reference_face_image
                       )
        return (images,fps,debug(time_sec))

    def process_video2(self):
        from facefusion.content_analyser import analyse_video
        from facefusion import wording, logger, state_manager, process_manager
        from facefusion.temp_helper import clear_temp_directory, create_temp_directory, get_temp_frame_paths
        from facefusion.vision import pack_resolution, restrict_video_resolution, unpack_resolution, restrict_video_fps
        from facefusion.processors.core import get_processors_modules
        from facefusion.ffmpeg import extract_frames
        for processor_module in get_processors_modules(state_manager.get_item('processors')):
            if not processor_module.pre_process('output'):
                return None
        conditional_append_reference_faces()
        if analyse_video(state_manager.get_item('target_path'), state_manager.get_item('trim_frame_start'),
                         state_manager.get_item('trim_frame_end')):
            return None
        # clear temp
        logger.debug(wording.get('clearing_temp'), __name__)
        clear_temp_directory(state_manager.get_item('target_path'))
        # create temp
        logger.debug(wording.get('creating_temp'), __name__)
        create_temp_directory(state_manager.get_item('target_path'))
        # extract frames
        process_manager.start()
        temp_video_resolution = pack_resolution(restrict_video_resolution(state_manager.get_item('target_path'),
                                                                          unpack_resolution(state_manager.get_item(
                                                                              'output_video_resolution'))))
        temp_video_fps = restrict_video_fps(state_manager.get_item('target_path'),
                                            state_manager.get_item('output_video_fps'))
        logger.info(wording.get('extracting_frames').format(resolution=temp_video_resolution, fps=temp_video_fps),
                    __name__)
        if extract_frames(state_manager.get_item('target_path'), temp_video_resolution, temp_video_fps):
            logger.debug(wording.get('extracting_frames_succeed'), __name__)
        else:
            process_manager.end()
            return None
        # process frames
        temp_frame_paths = get_temp_frame_paths(state_manager.get_item('target_path'))
        if temp_frame_paths:
            for processor_module in get_processors_modules(state_manager.get_item('processors')):
                logger.info(wording.get('processing'), processor_module.__name__)
                processor_module.process_video(state_manager.get_item('source_paths'), temp_frame_paths)
                processor_module.post_process()
        process_manager.end()

        from concurrent.futures import ThreadPoolExecutor
        from facefusion.vision import cv2imread
        frame_temp=[None]*len(temp_frame_paths)
        def func(idx,fname):
            frame_temp[idx] = torch.from_numpy(cv2imread(fname)[..., ::-1].copy())
        with ThreadPoolExecutor(max_workers=int(state_manager.get_item('execution_thread_count'))) as pool:
            for idx,fname in enumerate(temp_frame_paths):
                pool.submit(func,idx,fname)
        imgs = torch.stack(frame_temp) / 255.
        clear_temp_directory(state_manager.get_item('target_path'))
        return imgs

class WD_VIDEO2PATH:
    RETURN_TYPES = ("PATH",)
    RETURN_NAMES = ("path",)
    FUNCTION = "execute"
    CATEGORY = "WDTRIP"
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{"video":("VIDEO",)}}

    def execute(self,video):
        return (video,)


NODE_CLASS_MAPPINGS = {
    "WD_FaceFusion": WD_FaceFusion,
    "WD_FaceFusion_Video": WD_FaceFusion_Video,
    "WD_FaceFusion_Video2": WD_FaceFusion_Video2,
    "WD_VIDEO2PATH":WD_VIDEO2PATH,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WD_FaceFusion": "WD_FaceFusion",
    "WD_FaceFusion_Video": "WD_FaceFusion_Video",
    "WD_FaceFusion_Video2": "WD_FaceFusion_Video2",
    "WD_VIDEO2PATH":"WD_VIDEO2PATH"
}
