# in this script we download the data from oss and create symlink in the right place
import argparse
import os
import yaml
import hyperml

print('[prepare_data.py] - START')

parser = argparse.ArgumentParser(description='prepare training data')
parser.add_argument('-src', '--src_dir', help='source folder (datasets location, after download)', default='/data', type=str)
parser.add_argument('-dst', '--dst_dir', help='destination folder (where to put symbolic links)', default='data', type=str)
parser.add_argument('-rename', '--rename_dirs', type=yaml.load, default="{}")
args = parser.parse_args()

h = hyperml.init('hyperml.yaml')
# read credentials from environment variables
access_key = os.getenv('ALIYUN_ACCESS_KEY')
secret_key = os.getenv('ALIYUN_SECRET_KEY')
assert access_key is not None, 'should set env variable ALIYUN_ACCESS_KEY'
assert secret_key is not None, 'should set env variable ALIYUN_SECRET_KEY'
h.config['credentials']['oss']['default']['access_key_id'] = access_key
h.config['credentials']['oss']['default']['access_key_secret'] = secret_key
print('[download_data]: start')
datasets = h.config['assets']['oss']['datasets'].copy()
h.download_assets()
print('[download_data]: done')
print(h.config['assets']['oss']['datasets'])

# make symbolic link into "dst_dir"
# scan datasets
os.makedirs(args.dst_dir, exist_ok=True)
for key, value in datasets.items():
    assert len(value['keys']) == 1
    root_dir = value['local_dir']

    rel_path = value['keys'][0]['key']
    if rel_path[0] == '/':
        rel_path = rel_path[1:]
    if rel_path[-1] == '/':
        rel_path = rel_path[:-1]
    current_path = os.path.join(root_dir, rel_path)

    dir_name = os.path.split(current_path)[1]
    if key in args.rename_dirs:
        dir_name = args.rename_dirs[key]
    dst_path = os.path.join(args.dst_dir, dir_name)
    os.symlink(current_path, dst_path, target_is_directory=False)
    print('created symlink from [{}] to [{}]'.format(current_path, dst_path))

print('[prepare_data.py] - DONE')
