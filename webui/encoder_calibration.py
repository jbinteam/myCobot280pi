"""Paired-read encoder direction evidence, scoped to endpoint and firmware."""
import copy
import ipaddress
import json
import math
import os
from pathlib import Path
import tempfile
from webui.encoder_drag import direction_evidence


def context(host,port,firmware):
    host=str(ipaddress.IPv4Address(host))
    if (type(port) is not int or not 1<=port<=65535 or isinstance(firmware,bool)
            or not isinstance(firmware,(int,float)) or not math.isfinite(firmware) or firmware<=0):
        raise ValueError('Encoder calibration needs a verified robot endpoint and firmware')
    return host,port,float(firmware)


class EncoderCalibration:
    def __init__(self,path=None):
        self.path=Path(path) if path is not None else None
        self.records={};self.load_error=None
        if self.path is not None and self.path.exists():
            try:
                data=json.loads(self.path.read_text())
                if data.get('schema')!=1 or not isinstance(data.get('robots'),list):
                    raise ValueError('Unsupported calibration schema')
                for record in data['robots']:
                    key=context(record['host'],record['port'],record['firmware'])
                    if key in self.records:raise ValueError('Duplicate robot calibration')
                    self.records[key]=direction_evidence(record)
            except (OSError,ValueError,TypeError,KeyError,AttributeError) as exc:
                self.records={};self.load_error=f'Stored encoder calibration is invalid: {exc}'

    def lookup(self,host,port,firmware):
        if self.load_error:raise RuntimeError(self.load_error)
        evidence=self.records.get(context(host,port,firmware))
        return None if evidence is None else {'encoder_direction_evidence':copy.deepcopy(evidence)}

    def remember(self,host,port,firmware,mapping):
        if self.load_error:raise RuntimeError(self.load_error)
        key=context(host,port,firmware);fresh=direction_evidence(mapping)
        previous=self.records.get(key,[None]*6)
        if any(old is not None and new is not None and old['direction']!=new['direction']
               for old,new in zip(previous,fresh)):
            raise ValueError('Cannot merge contradictory encoder calibration')
        # Keep the original proof for known signs; fill newly observed joints.
        merged=[old if old is not None else new for old,new in zip(previous,fresh)]
        if not any(record is not None for record in merged) or merged==previous:return
        self.records[key]=merged;self._save()

    def forget(self,host,port,firmware):
        if self.records.pop(context(host,port,firmware),None) is not None:self._save()

    def _save(self):
        if self.path is None:return
        data={'schema':1,'robots':[{'host':host,'port':port,'firmware':firmware,
            'encoder_direction_evidence':evidence} for (host,port,firmware),evidence in sorted(self.records.items())]}
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(mode='w',dir=self.path.parent,prefix=self.path.name+'.',suffix='.tmp',delete=False) as out:
                temporary=Path(out.name);json.dump(data,out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
            temporary.replace(self.path)
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)
