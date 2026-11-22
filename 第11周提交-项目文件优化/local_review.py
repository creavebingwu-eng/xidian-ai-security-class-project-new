"""Select affected claims with paragraph neighbours and immutable evidence."""
import re

LOCAL_FORM='本次仅局部复审指定主张和相关上下文，不得给出整篇通过结论。核对上下文是否改变主张含义；缺失跨章节依赖时列入not_assessable并建议完整复审。其他输出结构与完整审阅相同。'

def select_context(draft,source,claim_ids):
    ids=list(dict.fromkeys(claim_ids))
    rows={m[1]:m[0] for m in re.finditer(r'^\s*\|\s*(C\d+)\s*\|.*$',draft,re.M)}
    if not ids or any(cid not in rows for cid in ids):raise ValueError('局部复审主张不存在')
    paragraphs=re.split(r'\n\s*\n',draft)
    keep=set(range(min(4,len(paragraphs))))
    for i,p in enumerate(paragraphs):
        if any(re.search(r'\b'+re.escape(cid)+r'\b',p) for cid in ids):
            keep.update(range(max(0,i-1),min(len(paragraphs),i+2)))
        if '证据边界' in p or '证据效力' in p:keep.add(i)
    chunks=[]
    for i in sorted(keep):
        lines=[line for line in paragraphs[i].splitlines() if not re.match(r'^\s*\|\s*C\d+\s*\|',line)]
        if lines:chunks.append('\n'.join(lines))
    chunks.append('# 本次局部复审主张\n\n'+'\n'.join(rows[cid] for cid in ids))
    evidence_ids=set()
    for cid in ids:evidence_ids.update(re.findall(r'\bE\d+\b',rows[cid]))
    headers=list(re.finditer(r'^Source note: .+\nSource note SHA-256: [^\n]+\n',source,re.M))
    selected=[]
    for n,header in enumerate(headers):
        body=source[header.end():headers[n+1].start() if n+1<len(headers) else len(source)]
        markers=list(re.finditer(r'^\[Evidence (E\d+) \|[^\n]+\]\n',body,re.M))
        units=[body[m.start():markers[i+1].start() if i+1<len(markers) else len(body)]
               for i,m in enumerate(markers) if m[1] in evidence_ids]
        # Keep source ordering for S-number mapping even when a source is unused.
        selected.append(header[0]+''.join(units))
    if not headers or not evidence_ids:raise ValueError('局部复审需要结构化Evidence与Source note')
    return '\n\n'.join(chunks), '\n'.join(selected), ids
