# hsy 2024.05.20 merge models import functions; merge configs

def import_functions_given_model_type(obj_think_model_type, img_caption_model_type):
    assert obj_think_model_type in ['gemini', 'claude', 'gpt4v', 'minigpt4', 'llava'], "Unsupported model type"
    print('verbose... model_type import_functions_given_obj_think_model_type {}'.format(obj_think_model_type))

    assert img_caption_model_type in ['gemini', 'claude', 'gpt4v', 'minigpt4', 'llava'], "Unsupported model type"
    print('verbose... model_type import_functions_given_img_caption_model_type {}'.format(img_caption_model_type))

    if obj_think_model_type == 'claude':
        from utils.utils_claude_clean import (
            generate_noun_given_scene_claude as generate_noun_given_scene_aimodel,
            random_obj_thinking_claude as random_obj_thinking_aimodel,
            irrelevant_obj_thinking_claude as irrelevant_obj_thinking_aimodel,
            filter_remove_obj_under_scene_claude as filter_remove_obj_under_scene_aimodel,
            filter_most_irrelevant_claude as filter_most_irrelevant_aimodel,
            correlated_obj_thinking_claude as correlated_obj_thinking_aimodel,
            correlated_example_create_claude as correlated_example_create_aimodel
        )

    elif obj_think_model_type == 'gemini':
        from utils.utils_gemini_clean import (
            generate_noun_given_scene_gemini as generate_noun_given_scene_aimodel,
            random_obj_thinking_gemini as random_obj_thinking_aimodel,
            irrelevant_obj_thinking_gemini as irrelevant_obj_thinking_aimodel,
            filter_remove_obj_under_scene_gemini as filter_remove_obj_under_scene_aimodel,
            filter_most_irrelevant_gemini as filter_most_irrelevant_aimodel,
            correlated_obj_thinking_gemini as correlated_obj_thinking_aimodel,
            correlated_example_create_gemini as correlated_example_create_aimodel
        )

    elif obj_think_model_type == 'gpt4v':
        from utils.utils_gpt_clean import (
            generate_noun_given_scene_gpt4v as generate_noun_given_scene_aimodel,
            random_obj_thinking_gpt4v as random_obj_thinking_aimodel,
            irrelevant_obj_thinking_gpt4v as irrelevant_obj_thinking_aimodel,
            filter_remove_obj_under_scene_gpt4v as filter_remove_obj_under_scene_aimodel,
            filter_most_irrelevant_gpt4v as filter_most_irrelevant_aimodel,
            correlated_obj_thinking_gpt4v as correlated_obj_thinking_aimodel,
            correlated_example_create_gpt4v as correlated_example_create_aimodel
        )

    elif obj_think_model_type == 'llava':
        from utils.utils_llava_clean import (
            generate_noun_given_scene_llava as generate_noun_given_scene_aimodel,
            random_obj_thinking_llava as random_obj_thinking_aimodel,
            irrelevant_obj_thinking_llava as irrelevant_obj_thinking_aimodel,
            filter_remove_obj_under_scene_llava as filter_remove_obj_under_scene_aimodel,
            filter_most_irrelevant_llava as filter_most_irrelevant_aimodel,
            correlated_obj_thinking_llava as correlated_obj_thinking_aimodel,
            correlated_example_create_llava as correlated_example_create_aimodel
        )

    elif obj_think_model_type == 'minigpt4':
        from utils.utils_minigpt4_clean import (
            generate_noun_given_scene_minigpt4 as generate_noun_given_scene_aimodel,
            random_obj_thinking_minigpt4 as random_obj_thinking_aimodel,
            irrelevant_obj_thinking_minigpt4 as irrelevant_obj_thinking_aimodel,
            filter_remove_obj_under_scene_minigpt4 as filter_remove_obj_under_scene_aimodel,
            filter_most_irrelevant_minigpt4 as filter_most_irrelevant_aimodel,
            correlated_obj_thinking_minigpt4 as correlated_obj_thinking_aimodel,
            correlated_example_create_minigpt4 as correlated_example_create_aimodel,
        )

    else:
        raise NotImplementedError(f'model type {obj_think_model_type} not implemented!')

    if img_caption_model_type == 'claude':
        from utils.utils_claude_clean import (
            gt_generation_claude as gt_generation_aimodel,
            gt_generation_multi_obj_removal_claude as gt_generation_multi_obj_removal_aimodel,
            image_caption_claude as image_caption_aimodel,
            list_objects_given_img_claude as list_objects_given_img_aimodel,
            vqa_claude as vqa_aimodel,
            safe_remove_dir as safe_remove_dir,
            close_logger as close_logger
        )

    elif img_caption_model_type == 'gemini':
        from utils.utils_gemini_clean import (
            gt_generation_gemini as gt_generation_aimodel,
            gt_generation_multi_obj_removal_gemini as gt_generation_multi_obj_removal_aimodel,
            image_caption_gemini as image_caption_aimodel,
            list_objects_given_img_gemini as list_objects_given_img_aimodel,
            vqa_gemini as vqa_aimodel,
            safe_remove_dir as safe_remove_dir,
            close_logger as close_logger
        )

    elif img_caption_model_type == 'gpt4v':
        from utils.utils_gpt_clean import (
            gt_generation_gpt4v as gt_generation_aimodel,
            gt_generation_multi_obj_removal_gpt4v as gt_generation_multi_obj_removal_aimodel,
            image_caption_gpt4v as image_caption_aimodel,
            list_objects_given_img_gpt4v as list_objects_given_img_aimodel,
            vqa_gpt4v as vqa_aimodel,
            safe_remove_dir as safe_remove_dir,
            close_logger as close_logger
        )

    elif img_caption_model_type == 'llava':
        from utils.utils_llava_clean import (
            gt_generation_llava as gt_generation_aimodel,
            gt_generation_multi_obj_removal_llava as gt_generation_multi_obj_removal_aimodel,
            image_caption_llava as image_caption_aimodel,
            list_objects_given_img_llava as list_objects_given_img_aimodel,
            vqa_llava as vqa_aimodel,
            safe_remove_dir as safe_remove_dir,
            close_logger as close_logger
        )

    elif img_caption_model_type == 'minigpt4':
        from utils.utils_minigpt4_clean import (
            gt_generation_minigpt4 as gt_generation_aimodel,
            gt_generation_multi_obj_removal_minigpt4 as gt_generation_multi_obj_removal_aimodel,
            image_caption_minigpt4 as image_caption_aimodel,
            list_objects_given_img_minigpt4 as list_objects_given_img_aimodel,
            vqa_minigpt4 as vqa_aimodel,
            safe_remove_dir as safe_remove_dir,
            close_logger as close_logger
        )

    else:
        raise NotImplementedError(f'model type {img_caption_model_type} not implemented!')

    return (
        generate_noun_given_scene_aimodel,
        random_obj_thinking_aimodel,
        irrelevant_obj_thinking_aimodel,
        gt_generation_aimodel,
        gt_generation_multi_obj_removal_aimodel,
        image_caption_aimodel,
        vqa_aimodel,
        filter_remove_obj_under_scene_aimodel,
        filter_most_irrelevant_aimodel,
        list_objects_given_img_aimodel,
        correlated_obj_thinking_aimodel,
        correlated_example_create_aimodel,
        safe_remove_dir,
        close_logger
    )

def load_cfg_given_model_type(model_type):
    assert model_type in ['gemini', 'claude', 'gpt4v', 'minigpt4', 'llava'], "Unsupported model type"

    if model_type == 'claude':
        temp_min = 0.0
        temp_max = 1.0

    elif model_type == 'gemini':
        temp_min = 0.0
        temp_max = 1.0

    elif model_type == 'gpt4v':
        temp_min = 0.2
        temp_max = 1.5

    elif model_type == 'llava':
        temp_min = 0.0
        temp_max = 1.0

    elif model_type == 'minigpt4':
        temp_min = 0.0
        temp_max = 1.0

    else:
        raise NotImplementedError(f'model type {model_type} not implemented!')

    temp_generate_noun_given_scene = temp_min
    temp_filter_remove_obj_under_scene = temp_max
    temp_filter_most_irrelevant = temp_min
    temp_random_obj_thinking = temp_max
    temp_irrelevant_obj_thinking = temp_max
    temp_correlated_obj_thinking = temp_max

    print(
        'verbose...load_cfg_given_obj_think_model_type given model_type {}: temp_generate_noun_given_scene {}, temp_filter_remove_obj_under_scene {}, temp_filter_most_irrelevant {}, temp_random_obj_thinking {}, temp_irrelevant_obj_thinking {}, temp_correlated_obj_thinking {}'.format(
            model_type, temp_generate_noun_given_scene, temp_filter_remove_obj_under_scene, temp_filter_most_irrelevant,
            temp_random_obj_thinking, temp_irrelevant_obj_thinking, temp_correlated_obj_thinking))

    return (temp_generate_noun_given_scene, temp_filter_remove_obj_under_scene, temp_filter_most_irrelevant,
            temp_random_obj_thinking, temp_irrelevant_obj_thinking, temp_correlated_obj_thinking)