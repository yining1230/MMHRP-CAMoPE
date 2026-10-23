"""Consolidated CAMoPE explanation and routing analysis for AHO.

Run --help without loading the scientific Python dependencies. Model weights
are required for inference; historical figures are not regenerated on import.
All output runs use new directories. No existing experiment is overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime

DATASET = 'AHO'


def find_root():
    for path in Path(__file__).resolve().parents:
        if (path / 'models/DeepLearnModel.py').is_file() and (path / 'data').is_dir():
            return path
    raise FileNotFoundError('Cannot locate project root; use the complete release directory.')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True, help='Trusted state_dict checkpoint; not included in this source release.')
    p.add_argument('--data', type=Path, help='Override the bundled dataset.')
    p.add_argument('--vocab', type=Path, help='Override the bundled SMILES vocabulary.')
    p.add_argument('--sample-index', type=int, default=1092 if DATASET == 'AHO' else 3116,
                   help='Zero-based position AFTER shuffling, not the source row identifier.')
    p.add_argument('--shuffle-seed', type=int, default=1 if DATASET == 'AHO' else 5)
    p.add_argument('--embedding-seed', type=int, default=5,
                   help='Seed for random embeddings of out-of-vocabulary SMILES tokens.')
    p.add_argument('--device', default='cpu', help='cpu, cuda, or cuda:N')
    p.add_argument('--tasks', nargs='+', default=['sample', 'conditions', 'routing', 'sankey', 'atoms', 'fragments', 'text'],
                   choices=['sample', 'conditions', 'routing', 'sankey', 'atoms', 'fragments', 'text', 'predictions', 'all'])
    p.add_argument('--max-samples', type=int, default=1000, help='Population analyses: 0 = full dataset; otherwise first N shuffled reactions.')
    p.add_argument('--bins', type=int, default=6)
    p.add_argument('--grid-size', type=int, default=30)
    p.add_argument('--temperature-range', type=float, nargs=2, metavar=('MIN', 'MAX'))
    p.add_argument('--secondary-range', type=float, nargs=2, metavar=('MIN', 'MAX'), help='Pressure (AHO) or time (Heck).')
    p.add_argument('--frame-step', type=int, default=50)
    p.add_argument('--output-dir', type=Path, help='Must not exist. Default: a new timestamped directory beside this script.')
    p.add_argument('--mark-original', action='store_true')
    p.add_argument('--axis-font', type=float, default=16)
    p.add_argument('--tick-font', type=float, default=14)
    p.add_argument('--legend-font', type=float, default=14)
    p.add_argument('--font-family', default='DejaVu Sans', help='Use Arial only if installed.')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--threads', type=int, default=1)
    a = p.parse_args()
    for name in ('bins', 'grid_size', 'frame_step', 'threads'):
        if getattr(a, name) < 1:
            p.error(f'--{name.replace("_", "-")} must be positive')
    if a.grid_size < 2 or a.max_samples < 0:
        p.error('grid-size must be at least 2; max-samples must be nonnegative')
    for value in (a.temperature_range, a.secondary_range):
        if value is not None and (not all(__import__('math').isfinite(x) for x in value) or value[0] >= value[1]):
            p.error('Each supplied range must have finite MIN < MAX')
    return a


def load_dependencies():
    global np, pd, torch, plt, Batch, Chem, BRICS, Draw
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from torch_geometric.data import Batch
    from rdkit import Chem
    from rdkit.Chem import BRICS, Draw


class Analysis:
    def __init__(self, args):
        self.args = args
        self.root = find_root()
        sys.path.insert(0, str(self.root))
        from models.DeepLearnModel import MMHRP_GCL_tri_MoPE_pdall
        from utils.molecule import smis_to_graph, smi_tokenizer
        from utils.rxn import df_to_AHO_rxn_list, df_to_rxn_list, get_AHO_RxnSmi, get_Heck_RxnSmi, vocab_txt_to_dict
        self.make_graph = smis_to_graph
        self.tokenizer = smi_tokenizer
        self.device = torch.device(args.device)
        torch.set_num_threads(args.threads)
        np.random.seed(args.embedding_seed)
        torch.manual_seed(args.embedding_seed)
        data_default = 'data/AHO/2AHO_all_condition.xlsx' if DATASET == 'AHO' else 'data/Heck/Heck_data_clean.xlsx'
        vocab_default = 'models/AHO_vocab.txt' if DATASET == 'AHO' else 'models/HeckLit_vocab.txt'
        self.data_path = (args.data or self.root / data_default).resolve()
        self.vocab_path = (args.vocab or self.root / vocab_default).resolve()
        table = pd.read_excel(self.data_path) if self.data_path.suffix.lower() != '.csv' else pd.read_csv(self.data_path)
        self.table = table.sample(random_state=args.shuffle_seed, frac=1).reset_index(drop=True)
        self.reactions = (df_to_AHO_rxn_list if DATASET == 'AHO' else df_to_rxn_list)(self.table)
        if not 0 <= args.sample_index < len(self.reactions):
            raise IndexError(f'sample-index outside [0, {len(self.reactions)-1}]')
        get_smiles = get_AHO_RxnSmi if DATASET == 'AHO' else get_Heck_RxnSmi
        raw = [get_smiles(r) for r in self.reactions]
        # Historical checkpoints were trained using maximum raw SMILES character
        # length for padding. Keep this convention, even though tokens are shorter.
        self.max_len = max(map(len, raw))
        self.tokens = [smi_tokenizer(s) for s in raw]
        self.vocab = vocab_txt_to_dict(str(self.vocab_path))
        self.conditions = np.asarray([[r.temp, r.pres, r.s] if DATASET == 'AHO' else [r.temp, r.time] for r in self.reactions], dtype=np.float32)
        if not np.isfinite(self.conditions).all():
            raise ValueError('Conditions contain missing or nonfinite values; inspect the input table.')
        self.reference = torch.tensor(self.conditions.mean(axis=0), dtype=torch.float32)
        self.conditions_names = ['temperature_C', 'pressure_atm', 'substrate_catalyst_ratio'] if DATASET == 'AHO' else ['temperature_C', 'time_h']
        self.target_scale = 1.0 if DATASET == 'AHO' else 100.0
        channels = [32, 64] if DATASET == 'AHO' else [16, 32]
        decoder = [1000, 500, 100] if DATASET == 'AHO' else [128, 64, 32]
        self.model = MMHRP_GCL_tri_MoPE_pdall(
            GraphEncoder={'NodeFeatNum': 7, 'Channels': channels, 'Heads': 4},
            TextEncoder={'SmiFeatNum': 128, 'Heads': 4 if DATASET == 'AHO' else 2, 'BigruChannels': [128,128], 'BigruNumlayer': 1},
            ConditionEncoder={'InputDim': self.conditions.shape[1], 'HiddenDim': 64, 'OutDim': 32},
            Decoder={'Channels': decoder}, emb_size=128, device=self.device, FDS=None)
        try:
            state = torch.load(args.checkpoint, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(args.checkpoint, map_location=self.device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        if not isinstance(state, dict):
            raise TypeError('Expected a state_dict, not a serialized model or scalar.')
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        for module in self.model.modules():
            if hasattr(module, 'device'):
                module.device = self.device
        self.output = args.output_dir or Path(__file__).resolve().parent / ('results_1023_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        if self.output.resolve() == self.root or self.output.exists():
            raise FileExistsError('Choose an output directory that does not exist.')
        self.output.mkdir(parents=True, exist_ok=False)
        plt.rcParams.update({'font.family': args.font_family, 'axes.labelsize': args.axis_font,
                             'xtick.labelsize': args.tick_font, 'ytick.labelsize': args.tick_font,
                             'legend.fontsize': args.legend_font})
        self.n_population = min(args.max_samples or len(self.reactions), len(self.reactions))
        self.population = None
        self.metadata = dict(dataset=DATASET, source_data=str(self.data_path), data_sha256=self.digest(self.data_path),
            checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=self.digest(args.checkpoint),
            vocabulary_sha256=self.digest(self.vocab_path), shuffle_seed=args.shuffle_seed,
            embedding_seed=args.embedding_seed, sample_index_after_shuffle=args.sample_index,
            source_reaction_id=str(self.reactions[args.sample_index].index), dataset_size=len(self.reactions),
            population_sample_count=self.n_population, condition_order=self.conditions_names,
            original_conditions=self.conditions[args.sample_index].tolist(),
            max_smiles_padding_length=self.max_len, model_training=False, settings=vars(args).copy(),
            limitations=['Routing weights are not predictive performance contributions.',
                'Condition scans are model predictions, not experimental measurements.',
                'Global marginal ranges do not guarantee joint training support.',
                'Checkpoint training/test membership is not known without a saved split manifest.',
                'Historical unrecorded random OOV embeddings cannot be reconstructed exactly.',
                'Population bin means do not isolate the causal effect of a single condition.'])

    @staticmethod
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def save_json(self, name, value):
        with (self.output / name).open('x', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=True, indent=2, default=lambda x: str(x) if isinstance(x, Path) else x.item())

    def sample(self, index):
        r = self.reactions[index]
        vectors = np.zeros((self.max_len, 128), dtype=np.float32)
        # Each sample has its own deterministic RNG: selection/order of tasks
        # cannot silently change the embeddings used for the same sample.
        rng = np.random.RandomState((self.args.embedding_seed + index) % (2**32))
        for i, token in enumerate(self.tokens[index]):
            vector = self.vocab.get(token)
            if vector is None:
                vector = rng.random_sample(128) * 0.5
                vector -= vector.mean()
            vectors[i] = vector
        chemicals = r.solvents + r.cats + ([] if DATASET == 'AHO' else r.reagents)
        return [self.make_graph(r.reactants + r.products, ChargeContribution=False),
                self.make_graph(chemicals, ChargeContribution=False), torch.tensor(vectors),
                torch.tensor(self.conditions[index]), float(r.rxn_ddg if DATASET == 'AHO' else r.rxn_yield / 100)]

    def tensors(self, sample):
        return [Batch.from_data_list([sample[0]]).to(self.device),
                Batch.from_data_list([sample[1]]).to(self.device),
                sample[2].unsqueeze(0).to(self.device), sample[3].unsqueeze(0).to(self.device)]

    def predict(self, sample):
        with torch.no_grad():
            return float(self.model(self.tensors(sample)).item())

    def weights(self, sample):
        with torch.no_grad():
            g1,g2,text,cond = self.tensors(sample)
            graph = torch.cat([self.model.ReaProEncoder(g1), self.model.CatSolEncoder(g2)], dim=1)
            w = self.model.router(torch.cat([graph, self.model.RxnSmiEncoder(text), self.model.CondEncoder(cond)], dim=1))
        w = w.cpu().numpy()[0]
        if not np.isfinite(w).all() or not np.isclose(w.sum(), 1.0, atol=1e-6):
            raise ValueError('Invalid Softmax routing weights.')
        return w

    def sample_result(self):
        s = self.sample(self.args.sample_index)
        pred = self.predict(s)
        self.save_json('sample_prediction.json', dict(sample_index_after_shuffle=self.args.sample_index,
            source_reaction_id=str(self.reactions[self.args.sample_index].index), conditions=s[3].tolist(),
            condition_order=self.conditions_names, observed=s[4]*self.target_scale,
            predicted=pred*self.target_scale, absolute_error=abs(pred-s[4])*self.target_scale,
            unit='kcal/mol' if DATASET == 'AHO' else 'yield percentage points', expert_weights=self.weights(s).tolist()))

    def color_map(self, grid, temperatures, secondary, name, label, vmin=None, vmax=None):
        fig,ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(grid, origin='lower', aspect='auto', cmap='viridis',
            extent=[secondary.min(), secondary.max(), temperatures.min(), temperatures.max()], vmin=vmin, vmax=vmax)
        ax.set_xlabel('Pressure (atm)' if DATASET == 'AHO' else 'Time (h)')
        ax.set_ylabel('Temperature (°C)')
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(label, fontsize=self.args.tick_font)
        cb.ax.tick_params(labelsize=self.args.tick_font)
        if self.args.mark_original:
            c = self.conditions[self.args.sample_index]
            ax.scatter(c[1],c[0],marker='*',s=150,c='white',edgecolors='black',label='Original condition',zorder=5)
            ax.legend()
        fig.tight_layout()
        fig.savefig(self.output / name, dpi=self.args.dpi)
        plt.close(fig)

    def condition_maps(self):
        s = self.sample(self.args.sample_index)
        a = self.args.temperature_range or [float(self.conditions[:,0].min()),float(self.conditions[:,0].max())]
        b = self.args.secondary_range or [float(self.conditions[:,1].min()),float(self.conditions[:,1].max())]
        if a[0] == a[1] or b[0] == b[1]:
            raise ValueError('Automatic scan bounds have zero width; supply explicit --temperature-range and --secondary-range.')
        temperatures, secondary = np.linspace(*a,self.args.grid_size), np.linspace(*b,self.args.grid_size)
        prediction = np.empty((len(temperatures),len(secondary)))
        routing = np.empty((*prediction.shape,self.model.num_experts))
        for i,t in enumerate(temperatures):
            for j,v in enumerate(secondary):
                changed = list(s)
                changed[3] = s[3].clone()
                changed[3][0],changed[3][1] = float(t),float(v)
                prediction[i,j],routing[i,j] = self.predict(changed)*self.target_scale,self.weights(changed)
            print(f'Condition scan row {i+1}/{len(temperatures)}', flush=True)
        pd.DataFrame(prediction,index=temperatures,columns=secondary).to_csv(self.output/'condition_prediction_grid.csv',index_label='temperature_C')
        label = r'Predicted $\Delta\Delta G^{\ddag}$ (kcal/mol)' if DATASET == 'AHO' else 'Predicted Yield (%)'
        self.color_map(prediction,temperatures,secondary,'condition_heatmap.png',label)
        for e in range(self.model.num_experts):
            values = routing[:,:,e]
            pd.DataFrame(values,index=temperatures,columns=secondary).to_csv(self.output/f'expert_{e}_condition_grid.csv',index_label='temperature_C')
            self.color_map(values,temperatures,secondary,f'expert_{e}_condition_heatmap.png',f'Expert {e} routing weight',float(routing.min()),float(routing.max()))
        self.metadata['condition_scan'] = dict(temperature_range=a,secondary_range=b,
            prediction_range=[float(prediction.min()),float(prediction.max())],
            fixed_other_conditions=s[3][2:].tolist(), expert_shared_scale=[float(routing.min()),float(routing.max())])
        if DATASET != 'AHO' and (prediction.min()<0 or prediction.max()>100):
            print('NOTE: Unconstrained regressor produced values outside 0-100%; values are preserved, not clipped.')

    def population_values(self):
        if self.population is None:
            weights,predictions = [],[]
            for i in range(self.n_population):
                s = self.sample(i)
                weights.append(self.weights(s))
                predictions.append(self.predict(s))
                if (i+1)%50 == 0:
                    print(f'Population inference {i+1}/{self.n_population}',flush=True)
            self.population = np.asarray(weights),np.asarray(predictions)
        return self.population

    @staticmethod
    def binned_average(x, weights, bins):
        x = np.asarray(x)
        edges = np.linspace(float(x.min()),float(x.max()),bins+1)
        if edges[0] == edges[-1]:
            return np.asarray([edges[0]]),np.mean(weights,axis=0,keepdims=True),np.asarray([len(x)]),np.asarray([edges[0],edges[-1]])
        ids = np.digitize(x,edges[1:-1])
        counts = np.bincount(ids,minlength=bins)
        means = np.asarray([weights[ids==i].mean(axis=0) if counts[i] else np.full(weights.shape[1],np.nan) for i in range(bins)])
        return (edges[:-1]+edges[1:])/2,means,counts,edges

    def routing_curves(self):
        weights,_ = self.population_values()
        for c,name in enumerate(self.conditions_names):
            centers,means,counts,edges = self.binned_average(self.conditions[:self.n_population,c],weights,self.args.bins)
            table = pd.DataFrame({'bin_center':centers,'n_samples':counts})
            for e in range(weights.shape[1]):
                table[f'expert_{e}_weight'] = means[:,e]
            table.to_csv(self.output/f'expert_vs_{name}.csv',index=False)
            fig,ax = plt.subplots(figsize=(6,4))
            for e in range(weights.shape[1]):
                ax.plot(centers,means[:,e],marker='o',label=f'Expert {e}')
            ax.set_xlabel(name.replace('_',' '))
            ax.set_ylabel('Average Expert Weight')
            ax.set_ylim(0,1)
            ax.legend(loc='center left')
            fig.tight_layout()
            fig.savefig(self.output/f'expert_vs_{name}.png',dpi=self.args.dpi)
            plt.close(fig)
            self.metadata[f'{name}_bins'] = dict(edges=edges.tolist(),counts=counts.tolist(),empty_bins='NaN; never interpreted as zero weight')
        pd.DataFrame({'expert':np.arange(weights.shape[1]),'mean_weight':weights.mean(0),
            'std_weight':weights.std(0),'top1_fraction':[(weights.argmax(1)==e).mean() for e in range(weights.shape[1])]}).to_csv(self.output/'expert_usage.csv',index=False)

    def predictions(self):
        _,pred = self.population_values()
        true = np.asarray([float(r.rxn_ddg if DATASET=='AHO' else r.rxn_yield/100) for r in self.reactions[:self.n_population]])
        pd.DataFrame({'sample_index_after_shuffle':np.arange(len(pred)),
            'source_reaction_id':[r.index for r in self.reactions[:len(pred)]],
            'observed':true*self.target_scale,'predicted':pred*self.target_scale,
            'absolute_error':np.abs(pred-true)*self.target_scale}).to_csv(self.output/'population_predictions.csv',index=False)

    def perturb(self,s,modality):
        changed = list(s)
        if modality == 0:
            for b in (0,1):
                changed[b] = s[b].clone()
                changed[b].x = torch.zeros_like(changed[b].x)
        elif modality == 1:
            changed[2] = torch.zeros_like(s[2])
        else:
            changed[3] = self.reference.clone()
        return changed

    @staticmethod
    def sankey_trace(matrix):
        import plotly.graph_objects as go
        return go.Sankey(arrangement='fixed',valueformat='.6f',
            node=dict(label=['Graph','Text','Numerical','Expert 0','Expert 1'],
                x=[0.03,0.03,0.03,0.94,0.94],y=[0.08,0.48,0.88,0.28,0.72],
                thickness=25,pad=25,color=['#9DBBDD','#F7C38B','#A9D6A5','#C9ADD0','#C9ADD0']),
            link=dict(source=[0,0,1,1,2,2],target=[3,4,3,4,3,4],value=matrix.ravel().tolist(),
                color=['rgba(103,151,197,0.34)']*2+['rgba(237,157,75,0.34)']*2+['rgba(91,170,88,0.34)']*2))

    def sankey(self):
        import plotly.graph_objects as go
        weights,_ = self.population_values()
        deltas = np.empty((self.n_population,3,weights.shape[1]))
        for i in range(self.n_population):
            s = self.sample(i)
            deltas[i] = [weights[i]-self.weights(self.perturb(s,m)) for m in range(3)]
        if weights.shape[1] != 2:
            raise ValueError('The published Sankey layout currently expects exactly two experts.')
        positive = np.maximum(deltas,0).mean(0)
        np.savez_compressed(self.output/'directional_routing_raw.npz',directional_deltas=deltas,base_weights=weights,positive_support=positive)
        pd.DataFrame(positive,index=['Graph','Text','Numerical'],columns=['Expert 0','Expert 1']).to_csv(self.output/'directional_routing_support.csv')
        pd.DataFrame(deltas.mean(0),index=['Graph','Text','Numerical'],columns=['Expert 0','Expert 1']).to_csv(self.output/'signed_routing_changes.csv')
        figure = go.Figure(self.sankey_trace(positive))
        figure.update_layout(font=dict(size=22),width=1200,height=700)
        figure.write_html(self.output/'sankey_static.html',include_plotlyjs=True)
        # Matplotlib static output does not require Kaleido or a browser runtime.
        fig,ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(positive,aspect='auto',cmap='Blues')
        ax.set_xticks([0,1]);ax.set_xticklabels(['Expert 0','Expert 1'])
        ax.set_yticks([0,1,2]);ax.set_yticklabels(['Graph','Text','Numerical'])
        for m in range(3):
            for e in range(2):
                ax.text(e,m,f'{positive[m,e]:.4f}',ha='center',va='center',color='black')
        fig.colorbar(im,ax=ax,label='Mean positive routing shift')
        fig.tight_layout();fig.savefig(self.output/'routing_support_matrix.png',dpi=self.args.dpi);plt.close(fig)
        step = self.args.frame_step
        ends = sorted(set(list(range(step,self.n_population+1,step))+[self.n_population]))
        for mode in ('cumulative','blocks'):
            matrices,labels = [],[]
            for k,end in enumerate(ends):
                start = 0 if mode=='cumulative' else (ends[k-1] if k else 0)
                matrices.append(np.maximum(deltas[start:end],0).mean(0))
                labels.append(f'N={end}' if mode=='cumulative' else f'{start+1}-{end}')
            frames = [go.Frame(name=l,data=[self.sankey_trace(v)]) for l,v in zip(labels,matrices)]
            dynamic = go.Figure(data=[self.sankey_trace(matrices[0])],frames=frames)
            dynamic.update_layout(font=dict(size=22),width=1200,height=750,
                title=f'{DATASET}: {mode} mean positive routing support',
                updatemenus=[dict(type='buttons',buttons=[dict(label='Play',method='animate',args=[None,{'frame':{'duration':650,'redraw':True}}]),
                    dict(label='Pause',method='animate',args=[[None],{'mode':'immediate','frame':{'duration':0,'redraw':False}}])])],
                sliders=[dict(steps=[dict(label=l,method='animate',args=[[l],{'mode':'immediate','frame':{'duration':0,'redraw':True}}]) for l in labels])])
            dynamic.write_html(self.output/f'sankey_dynamic_{mode}.html',include_plotlyjs=True,auto_play=False)
        self.metadata['sankey'] = dict(formula='mean(max(original_weight - perturbed_weight, 0))',
            graph_reference='zero node features in both branches; topology preserved',
            text_reference='zero precomputed SMILES embedding sequence',
            numerical_reference=self.reference.tolist(),reference_population='entire supplied dataset',
            interpretation='Directional routing support, not accuracy contribution; expert indices are model-specific.')

    def atom_scores(self,s,branch,groups=None):
        baseline = self.predict(s)
        if groups is None:
            groups = [(i,) for i in range(s[branch].num_nodes)]
        scores=[]
        for group in groups:
            changed=list(s)
            changed[branch]=s[branch].clone()
            changed[branch].x[list(group)]=0
            scores.append((baseline-self.predict(changed))*self.target_scale)
        return np.asarray(scores)

    def molecule(self,index,branch):
        r=self.reactions[index]
        smis=r.reactants+r.products if branch==0 else r.solvents+r.cats+([] if DATASET=='AHO' else r.reagents)
        mol=Chem.MolFromSmiles('.'.join(smis))
        if mol is None:
            raise ValueError('Cannot parse component SMILES.')
        return mol

    def draw_atoms(self,mol,scores,name):
        max_score=max(float(np.abs(scores).max()),1e-12)
        colors={i:(1.0,1.0-float(abs(v))/max_score,1.0-float(abs(v))/max_score) for i,v in enumerate(scores)}
        drawer=Draw.MolDraw2DCairo(1600,650)
        drawer.DrawMolecule(mol,highlightAtoms=list(colors),highlightAtomColors=colors)
        drawer.FinishDrawing()
        (self.output/name).write_bytes(drawer.GetDrawingText())

    def atoms(self):
        s=self.sample(self.args.sample_index)
        for branch in (0,1):
            mol=self.molecule(self.args.sample_index,branch)
            if mol.GetNumAtoms()!=s[branch].num_nodes:
                raise ValueError('Atom-to-graph node mapping mismatch; attribution stopped.')
            scores=self.atom_scores(s,branch)
            pd.DataFrame({'atom_index':np.arange(len(scores)),'element':[a.GetSymbol() for a in mol.GetAtoms()],
                'signed_prediction_change':scores,'absolute_prediction_change':np.abs(scores)}).to_csv(self.output/f'branch_{branch}_atom_importance.csv',index=False)
            self.draw_atoms(mol,scores,f'branch_{branch}_atom_importance.png')

    def fragments(self):
        s=self.sample(self.args.sample_index)
        # Preserve the original notebooks' SMARTS definitions.
        smarts={'Phenyl':'c1ccccc1','Carbonyl':'[CX3]=O','Ester':'[CX3](=O)[OX2H0]',
            'Amide':'[NX3][CX3](=O)[#6]','Alcohol':'[OX2H]','Ether':'[OD2]([#6])[#6]',
            'Halogen':'[F,Cl,Br,I]','Alkene':'C=C','Alkyne':'C#C',
            'Nitro':'[NX3](=O)=O','Amine':'[NX3;H2,H1;!$(NC=O)]'}
        for branch in (0,1):
            mol=self.molecule(self.args.sample_index,branch)
            groups=[];labels=[];kinds=[]
            for name,pattern in smarts.items():
                for match in mol.GetSubstructMatches(Chem.MolFromSmarts(pattern)):
                    groups.append(tuple(match));labels.append(name);kinds.append('functional_group')
            # Breaking BRICS bonds preserves original atom indices. Dummy atoms
            # have indices beyond the original molecule and are excluded.
            broken=BRICS.BreakBRICSBonds(mol)
            for i,match in enumerate(Chem.GetMolFrags(broken,asMols=False)):
                atoms=tuple(a for a in match if a<mol.GetNumAtoms())
                if atoms:
                    groups.append(atoms);labels.append(f'BRICS_{i}');kinds.append('BRICS')
            scores=self.atom_scores(s,branch,groups)
            atom_changes=self.atom_scores(s,branch)
            aggregated=np.asarray([atom_changes[list(group)].mean() for group in groups])
            pd.DataFrame({'kind':kinds,'label':labels,'atom_indices':[' '.join(map(str,g)) for g in groups],
                'signed_prediction_change':scores,'absolute_prediction_change':np.abs(scores),
                'mean_single_atom_masking_change':aggregated}).to_csv(self.output/f'branch_{branch}_fragment_importance.csv',index=False)
            if len(scores):
                selected=np.argsort(np.abs(scores))[-20:]
                fig,ax=plt.subplots(figsize=(8,6))
                ax.barh(np.arange(len(selected)),scores[selected])
                ax.set_yticks(np.arange(len(selected)));ax.set_yticklabels([labels[i] for i in selected])
                ax.set_xlabel('Original - masked prediction')
                fig.tight_layout();fig.savefig(self.output/f'branch_{branch}_fragment_importance.png',dpi=self.args.dpi);plt.close(fig)

    def text(self):
        s=self.sample(self.args.sample_index)
        captured=[]
        # Observe the Transformer input without replacing its forward method.
        handle=self.model.trans.register_forward_pre_hook(lambda module,inputs:captured.append(inputs[0].detach()))
        try:
            before=self.predict(s)
        finally:
            handle.remove()
        layer=self.model.trans
        x=captured[0]
        q=layer.norm1(x) if layer.norm_first else x
        with torch.no_grad():
            try:
                _,attention=layer.self_attn(q,q,q,need_weights=True,average_attn_weights=False)
            except TypeError:
                _,attention=layer.self_attn(q,q,q,need_weights=True)
        if attention.dim()==4:
            attention=attention.mean(1)
        a=attention[0].cpu().numpy()
        tokens=self.tokens[self.args.sample_index]
        valid=len(tokens)
        a=a[:valid,:valid]
        # Cropped attention is NOT renormalized: the model itself attends to
        # padded embedding rows. Export original values, not a modified model.
        np.save(self.output/'text_attention.npy',a)
        pd.DataFrame({'token_index':np.arange(valid),'token':tokens,'mean_incoming_attention':a.mean(0)}).to_csv(self.output/'text_token_attention.csv',index=False)
        if not np.isclose(before,self.predict(s),atol=1e-7):
            raise RuntimeError('Prediction changed during attention extraction.')
        fig,ax=plt.subplots(figsize=(8,6))
        im=ax.imshow(a,aspect='auto',cmap='Reds')
        ax.set_xlabel('Key token position');ax.set_ylabel('Query token position')
        fig.colorbar(im,ax=ax,label='Mean attention weight')
        fig.tight_layout();fig.savefig(self.output/'text_attention.png',dpi=self.args.dpi);plt.close(fig)
        # Keep repeated token strings distinct by using positions as identities.
        selected=np.argsort(a.mean(0))[-min(30,valid):]
        fig,ax=plt.subplots(figsize=(10,6))
        ax.bar(np.arange(len(selected)),a.mean(0)[selected])
        ax.set_xticks(np.arange(len(selected)))
        ax.set_xticklabels([f'{i}:{tokens[i]}' for i in selected],rotation=90,fontsize=9)
        ax.set_ylabel('Mean incoming attention')
        fig.tight_layout();fig.savefig(self.output/'text_token_attention.png',dpi=self.args.dpi);plt.close(fig)
        edges=[]
        for i in range(valid):
            candidates=np.argsort(a[i])[::-1]
            for j in [int(j) for j in candidates if j!=i][:3]:
                edges.append(dict(query_position=i,key_position=j,query_token=tokens[i],key_token=tokens[j],weight=float(a[i,j])))
        pd.DataFrame(edges).to_csv(self.output/'text_top_token_dependencies.csv',index=False)
        self.metadata['attention'] = dict(method='Observe original Transformer input; evaluate its self-attention with original parameters and norm_first convention',
            original_forward_preserved=True, padding_crop_renormalized=False,
            interpretation='Descriptive attention associations; not causal chemical importance.')


def main():
    args=parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f'Checkpoint not found: {args.checkpoint}. Supply a trained state_dict; weights are not bundled.')
    os.environ.setdefault('OMP_NUM_THREADS',str(args.threads))
    load_dependencies()
    analysis=Analysis(args)
    tasks=set(args.tasks)
    if 'all' in tasks:
        tasks={'sample','conditions','routing','sankey','atoms','fragments','text','predictions'}
    methods={'sample':'sample_result','conditions':'condition_maps','routing':'routing_curves',
        'sankey':'sankey','atoms':'atoms','fragments':'fragments','text':'text','predictions':'predictions'}
    try:
        for task,method in methods.items():
            if task in tasks:
                print(f'Running {DATASET}: {task}',flush=True)
                getattr(analysis,method)()
        analysis.metadata['run_status']='complete'
    except Exception as e:
        analysis.metadata['run_status']='failed'
        analysis.metadata['failure']=f'{type(e).__name__}: {e}'
        raise
    finally:
        analysis.save_json('run_metadata.json',analysis.metadata)
    print(f'Saved results to {analysis.output}',flush=True)


if __name__=='__main__':
    main()
