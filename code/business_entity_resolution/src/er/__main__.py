import argparse
from pathlib import Path


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('Must be positive')
    return value


def main():
    parser = argparse.ArgumentParser(description='Offline, CPU-only business entity resolution')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('index', 'train', 'predict', 'validate', 'run', 'make-smoke', 'package'):
        p = sub.add_parser(command)
        p.add_argument('--dataset', type=Path, required=True, help='Directory containing train/ and test/')
        if command == 'make-smoke':
            p.add_argument('--destination', type=Path, required=True)
            p.add_argument('--entities', type=positive, default=1000)
            p.add_argument('--distractors', type=positive, default=15000, help='Per target source')
            continue
        p.add_argument('--work', type=Path, required=True, help='Persistent local index and model directory')
        if command == 'index':
            p.add_argument('--split', choices=['train', 'test'], required=True)
        if command in ('train', 'run'):
            p.add_argument('--sample-size', type=positive, default=20000)
            p.add_argument('--cap', type=positive, default=60)
            p.add_argument('--seed', type=int, default=42)
        if command in ('train', 'predict', 'run'):
            p.add_argument('--threads', type=positive, default=4)
        if command in ('predict', 'validate', 'run', 'package'):
            p.add_argument('--output', type=Path, required=True)
        if command == 'package':
            p.add_argument('--team', required=True)
            p.add_argument('--members', required=True)
            p.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'make-smoke':
        from .smoke import make_smoke
        make_smoke(args.dataset, args.destination, args.entities, args.distractors)
        return
    args.work.mkdir(parents=True, exist_ok=True)
    if args.command in ('index', 'run'):
        from .index import build_index
        from .io import source_paths
        split = args.split if args.command == 'index' else 'train'
        build_index(source_paths(args.dataset, split)[1:], args.work / f'{split}.sqlite')
    if args.command in ('train', 'run'):
        from .training import train
        train(args.dataset, args.work, args.sample_size, args.cap, args.threads, args.seed)
    if args.command == 'run':
        build_index(source_paths(args.dataset, 'test')[1:], args.work / 'test.sqlite')
    if args.command in ('predict', 'run'):
        from .inference import predict
        predict(args.dataset, args.work, args.output, args.threads)
    if args.command == 'validate':
        from .inference import validate
        validate(args.dataset, args.work, args.output)
    if args.command == 'package':
        from .packaging import package
        package(args.dataset, args.work, args.output, args.team, args.members, args.archive)


if __name__ == '__main__':
    main()
