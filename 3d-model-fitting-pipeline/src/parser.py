import sys
sys.path.append('../../../utils')
import libraries
from libraries import *
import utils
from utils import *

class BaseOptions():
    def __init__(self):

        # import argparse
        # parser = argparse.ArgumentParser()
        self.parser = argparse.ArgumentParser()

        # self.parser = argparse.ArgumentParser()
        # # self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # self.parser.add_argument('--help', type=str, default='function that given an image, \
        # compute landmarks, generate 3D shapes and texture maps and renders a range of profiles \
        # in an angle from -90 to 90 degrees wrt a frontal position')
        
        """
        Add package description and ussage:
        python main.py -i ../../video-processing/output_frames -compLan -comp3D -ss
        """
        
        self.parser.add_argument('-ds', type=str, required=False, default=None, help='input dir: path to dataset')
        self.parser.add_argument('-i', type=str, required=False, default=None, help='input dir: path to dataset')
        self.parser.add_argument('-s', action='store_true', help='input subdir: name of input dir subdirs (e.g.: trainset) -same structure will be create din the ouput dir')
        self.parser.add_argument('-ss', action='store_true', help='subdir subdir: name of subdir within subdir -same structure will be create din the ouput dir')
        self.parser.add_argument('-o', type=str, default=None, help='output dir: if not provided it will be created within the input dir')
        self.parser.add_argument('-l', type=str, default=None, help='landmarks dir: if -comp3D is enabled; and landmarks are not stored in the <output dir>/landmarks')
        self.parser.add_argument('-tmap_size', type=int, default=512, help='texture map size: if -comp3D is enabled; image size of the output texture map')
        self.parser.add_argument('-step_size', type=int, default=10, help='step size: if RendPoses=True degrees step for rendering the profile rendering is performed')
        self.parser.add_argument('-mirror', choices=[True, False, 'only'], default=True, help='compute for mirror: if mirror image detected with file name "*_mirror.<ext>" compute functions')
        self.parser.add_argument('-z', type=int, default=1, dest='zoom', help='zoom factor: if -compLan is enabled; number of times the image is upsampled in order to detect faces')
        self.parser.add_argument('-sample', type=int, default=None, dest='sample_size', help='number of samples to compute')


        self.parser.add_argument('-compLan', action='store_true', help='compute landmarks')
        self.parser.add_argument('-comp3D', action='store_true', help='compute the 3D shape and texture map')
        self.parser.add_argument('-rendPos', action='store_true', help='render different poses using Mayavi software')

        # self.parser.add_argument('-cp_dir', type=str, help='checkpoint directory')
        # self.parser.add_argument('-run', type=str, default='', help='name for the iteration, e.g.: "lr_0.0002_bs_256"')

        # self.initialized = True

    def parse(self):
        if not self.initialized: self.initialize()
        self.opt = self.parser.parse_args()

        #setting INPUT/OUTPUT paths
        
        if self.opt.ds is not None: 
                self.opt.i = os.path.join('../../../datasets', self.opt.ds)
                self.opt.o = os.path.join(self.opt.i, 'output_fitting_pipeline')
        else:
                self.opt.o = os.path.join(self.opt.i.split(self.opt.i.split('/')[-1])[0], 'output_fitting_pipeline')

        # # for training models
        # if not self.opt.cp_dir: self.opt.cp_dir = 'checkpoint'
        # # save options details to the checkpoint folder
        # out_dirs = os.path.join(os.path.join(self.opt.o, self.opt.cp_dir), self.opt.run)
        # if not os.path.exists(out_dirs): os.makedirs(out_dirs)

        self.initialized = True
        args = vars(self.opt); print(args)

        print('------------ Options -------------')
        for k, v in sorted(args.items()): print('%s: %s' % (str(k), str(v)))
        print('----------------------------------')

        # f_name = os.path.join(out_dirs, 'opt.txt')
        # with open(f_name, 'wt') as opt_file:
        #     opt_file.write('------------ Options -------------\n')
        #     for k, v in sorted(args.items()): opt_file.write('%s: %s\n' % (str(k), str(v)))
        #     opt_file.write('----------------------------------\n')

        return self.opt

class FittingParser(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--show', type=bool, default=True, help='plot a sample image from the output folder')
