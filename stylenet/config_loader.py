import argparse
from typing import Union, Dict, Any, Optional
from pathlib import Path
import json
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

JSON_FORMAT = '.json'
YML_FORMAT = '.yml'


class HParams:
    '''
    Parser for configurations.
    '''
    def __init__(self, configs: Dict):
        for key, value in configs.items():
            setattr(self, key, value)

    def __str__(self):
        text = 'PARAMETERS\n' + '-'*20 + '\n'
        for (key, value) in self.__dict__.items():
            text += f'{key}:\t{value}' + '\n'

        return text

    @classmethod
    def load(cls, config_path: Union[str, Path]) -> 'HParams':
        '''
        parse config file of YML or JSON format.

        :param config_path: path to config file.
        :return dict of configurations.
        '''
        if isinstance(config_path, str):
            config_path = Path(config_path)
        assert config_path.exists()

        with config_path.open() as f:
            if config_path.suffix == JSON_FORMAT:
                configs = json.load(f)
            elif config_path.suffix == YML_FORMAT:
                configs = yaml.load(f, Loader=Loader)
            else:
                raise Exception('config_loader: config format is unknown.')

        return cls(configs)

    @classmethod
    def parse(cls, args: Union[Dict, 'argparse.Namespace']) -> 'HParams':
        '''
        parse arguments from dict or argumentparser

        :param args: parameters to save.
        '''
        if isinstance(args, argparse.Namespace):
            arguments = {}
            for (key, value) in vars(args).items():
                arguments[key] = value
        else:
            arguments = args

        return cls(arguments)

    def parse_and_add(
            self,
            args: Union[Dict, 'argparse.Namespace'],
            update: bool=True
    ) -> None:
        '''
        parse argument and add them.

        :param args: parameters to save.
        :param update: update value if it already exists.
        '''
        if isinstance(args, argparse.Namespace):
            arguments = {}
            for (key, value) in vars(args).items():
                arguments[key] = value
        else:
            arguments = args

        for (key, value) in arguments.items():
            self.add_hparam(key, value, update)

    def save(self, save_path: Union[str, Path]) -> None:
        '''
        save config parameters to save_path.

        :param save_path: path to save config parameters.
        '''
        if isinstance(save_path, str):
            save_path = Path(save_path)
        assert save_path.suffix in [JSON_FORMAT, YML_FORMAT], \
            'config_loader: config format is unknown.'

        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)

        with save_path.open('w') as f:
            if save_path.suffix == JSON_FORMAT:
                json.dump(self.__dict__, f)
            elif save_path.suffix == YML_FORMAT:
                f.write(yaml.dump(self.__dict__, default_flow_style=False))

    def add_hparam(self, key: str, value: Any, update: bool=True) -> None:
        '''
        add argument if key is not exists.
        it updates value when update=True
        '''
        if self.get(key) is not None:
            if update and self.get(key) != value:
                print(f'Update parameter {key}: {self.get(key)} -> {value}')
                setattr(self, key, value)
        else:
            setattr(self, key, value)

    def del_hparam(self, key: str) -> None:
        '''
        delete argument if key is exists.
        '''
        if hasattr(self, key):
            print(f'Delate parameter {key}')
            delattr(self, key)

    def get(self, key, default: Optional[Any]=None) -> Any:
        '''
        get value from key.
        '''
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return default
