import argparse
import os


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--help', type=str, default='function that given an image, \
        compute landmarks, generate 3D shapes and texture maps and renders a range of profiles \
        in an angle from -90 to 90 degrees wrt a frontal position')

        self.parser.add_argument('--input_path', required=True, help='path to images')
        self.parser.add_argument('--output_path', required=True, help='path to images (if the folder is not detected it will be created)')
        self.parser.add_argument('--texture_size', default=256, help='image size of the generated texture map')
        self.parser.add_argument('--step_size', default=10, help='degrees each which the profile rendering is performed')

        self.parser.add_argument('--computeLandmarks', type=bool, default=True, help='compute landmarks')
        self.parser.add_argument('--compute3D', type=bool, default=True, help='compute the 3D shape and texture map')
        self.parser.add_argument('--renderPoses', type=bool, default=False, help='render different poses using Mayavi software')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
