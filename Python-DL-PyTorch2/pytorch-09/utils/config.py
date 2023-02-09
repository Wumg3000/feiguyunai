from pprint import pprint


class Config:
    # data
    #voc_data_dir = 'D:/DataClinic/simple-faster-rcnn-pytorch/VOCdevkit/VOC2007/'
    voc_data_dir = 'C:/Users/wumg/jupyter-ipynb/pytorch-2-book/data/VOC2007/'
    min_size = 600  # image resize
    max_size = 1000  # image resize

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

config = Config()