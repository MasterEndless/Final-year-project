class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'CAER':
            root_dir = '/home/liuhan/C3D/CAER_Resize_datasets/'
            output_dir = '/home/liuhan/C3D/CAER_Resize_datasets/'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return './Models/ucf101-caffe.pth'