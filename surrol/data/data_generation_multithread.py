"""
Data generation for the case of Psm Envs and demonstrations.
Refer to
https://github.com/openai/baselines/blob/master/baselines/her/experiment/data_generation/fetch_data_generation.py
"""
import os
import argparse
import gym
import time
import numpy as np
import imageio
import surrol.gym
from surrol.const import ROOT_DIR_PATH
import multiprocessing

parser = argparse.ArgumentParser(description='generate demonstrations for imitation')
parser.add_argument('--env', type=str, required=True,
                    help='the environment to generate demonstrations')
parser.add_argument('--video', action='store_true',
                    help='whether or not to record video')
parser.add_argument('--steps', type=int,
                    help='how many steps allowed to run')
parser.add_argument('--processes', type=int, default=0,
                    help='number of processes/threads to run (default: cpu count)')
args = parser.parse_args()

actions = []
observations = []
infos = []

images = []  # record video
masks = []


def worker(task_id, num_itr, steps, video, env_name, queue=None):
    """Worker function to generate data in parallel."""
    try:
        local_actions = []
        local_observations = []
        local_infos = []
        local_images = []
        local_masks = []

        # print(f"Worker {task_id} starting...")
        env = gym.make(env_name, render_mode='human' if video else None)
        env.reset()

        for _ in range(num_itr):
            obs = env.reset()
            episode_acs, episode_obs, episode_info = [], [obs], []
            time_step, success = 0, False

            while time_step < steps:
                action = env.get_oracle_action(obs)
                if video:
                    img = env.render('rgb_array')
                    local_images.append(img)

                obs, reward, done, info = env.step(action)
                time_step += 1

                if isinstance(obs, dict) and info['is_success'] > 0 and not success:
                    success = True

                episode_acs.append(action)
                episode_info.append(info)
                episode_obs.append(obs)

            if success:
                local_actions.append(episode_acs)
                local_observations.append(episode_obs)
                local_infos.append(episode_info)
            
            if queue:
                queue.put(1)

        env.close()
        # print(f"Worker {task_id} finished.")
        return local_actions, local_observations, local_infos, local_images, local_masks
    except Exception as e:
        import traceback
        print(f"\nError in worker {task_id}: {e}")
        traceback.print_exc()
        return [], [], [], [], []

def main():
    env = gym.make(args.env, render_mode='human')  # 'human'
    num_itr = 200*100 if not args.video else 1
    cnt = 0
    init_state_space = 'random'
    env.reset()
    print("Reset!")
    init_time = time.time()

    if args.steps is None:
        args.steps = env._max_episode_steps

    print()
    # Determine number of processes
    if args.processes > 0:
        num_processes = args.processes
    else:
        num_processes = multiprocessing.cpu_count()
    
    print(f"Generating data with {num_processes} processes...")
    num_itr_per_process = int(np.ceil(num_itr / num_processes))

    manager = multiprocessing.Manager()
    queue = manager.Queue()

    with multiprocessing.Pool(processes=num_processes) as pool:
        tasks = [(i, num_itr_per_process, args.steps, args.video, args.env, queue) for i in range(num_processes)]
        result_async = pool.starmap_async(worker, tasks)
        
        total_tasks = num_itr_per_process * num_processes
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_tasks, desc="Generating Data")
        except ImportError:
            print(f"tqdm not installed. Progress: 0/{total_tasks}")
            pbar = None

        processed_count = 0
        while not result_async.ready():
            try:
                # Wait for a short time for an update
                queue.get(timeout=0.5)
                processed_count += 1
                if pbar:
                    pbar.update(1)
                else:
                    if processed_count % 10 == 0:
                        print(f"Progress: {processed_count}/{total_tasks}", end='\r')
            except:
                pass
        
        if pbar:
            pbar.close()
        elif processed_count > 0:
            print() # New line after text progress
            
        results = result_async.get()

    # Combine results from all processes
    for res in results:
        actions.extend(res[0])
        observations.extend(res[1])
        infos.extend(res[2])
        images.extend(res[3])
        masks.extend(res[4])

    file_name = "data_"
    file_name += args.env
    file_name += "_" + init_state_space
    file_name += "_" + str(num_itr)
    file_name += ".npz"

    folder = 'demo' if not args.video else 'video'
    folder = os.path.join(ROOT_DIR_PATH, 'data', folder)

    np.savez_compressed(os.path.join(folder, file_name),
                        acs=actions, obs=observations, info=infos)  # save the file

    if args.video:
        video_name = "video_"
        video_name += args.env + ".mp4"
        writer = imageio.get_writer(os.path.join(folder, video_name), fps=20)
        for img in images:
            writer.append_data(img)
        writer.close()

        if len(masks) > 0:
            mask_name = "mask_"
            mask_name += args.env + ".npz"
            np.savez_compressed(os.path.join(folder, mask_name),
                                masks=masks)  # save the file

    used_time = time.time() - init_time
    print("Saved data at:", folder)
    print("Time used: {:.1f}m, {:.1f}s\n".format(used_time // 60, used_time % 60))
    print(f"Trials: {num_itr}/{cnt}")
    env.close()


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
