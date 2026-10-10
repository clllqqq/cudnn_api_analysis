import re, sys, collections

HEADER = re.compile(r'^([IWE])! CuDNN \(v\d+ \d+\) function (cudnn[A-Za-z0-9_]*)\(\) called:')
STATUS = re.compile(r'^i!\s+status:')

def count(path):
    lines = open(path, encoding='utf-8', errors='replace').read().split('\n')
    heads = [i for i, l in enumerate(lines) if HEADER.match(l)]
    calls = collections.Counter()
    for j, i in enumerate(heads):
        end = heads[j + 1] if j + 1 < len(heads) else len(lines)
        payload = [b for b in lines[i + 1:end] if b.strip()]
        if HEADER.match(lines[i]).group(1) != 'I':
            continue
        if payload and STATUS.match(payload[0]):
            continue
        if not payload:
            continue
        calls[HEADER.match(lines[i]).group(2)] += 1
    return calls

if __name__ == '__main__':
    INT = {'cudnnGraphLibraryConfigInit', 'cudnnBackendExecuteInternal',
           'cudnnBackendFinalizeGraphVisualize', 'cudnnBackendFinalizeEngineCfgVisualize',
           'cudnnBackendExecuteGraphVisualize', 'cudnnCTCLossInternal',
           'cudnnGetCTCLossWorkspaceSizeInternal', 'cudnnGetProperty'}

    def internal(name):
        return (name in INT or name.endswith('Internal') or 'Visualize' in name
                or name.startswith('cudnnBackendGetAttribute_'))

    for path in sys.argv[1:]:
        c = count(path)
        pub = {k: v for k, v in c.items() if not internal(k)}
        print('== %s   公开入口合计 %d' % (path.split('/')[-1], sum(pub.values())))
        print('   ' + '、'.join('%s×%d' % (k.replace('cudnn', ''), v)
                              for k, v in sorted(pub.items(), key=lambda x: -x[1])))
