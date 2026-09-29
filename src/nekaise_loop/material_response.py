"""Recover actual Author teaching text; retain every adaptation beside raw evidence.

This is a serialization adapter, not a content critic or another model call. It
never fabricates an answer, guesses a citation, or trains provider/tool diagnostics.
"""
import json
import re

from .artifacts import digest
from .material_types import Candidate

WRAPPERS = ('rows', 'candidates', 'examples', 'materials', 'data')
PROMPTS = ('student_prompt', 'question', 'prompt', 'user_prompt', 'input')
ANSWERS = ('training_response', 'answer', 'response', 'completion', 'output')
TEXTS = ('training_text', 'text', 'content', 'prose')


class NoTrainingContent(ValueError):
    def __init__(self, audit):
        self.audit = audit
        super().__init__('Author returned no recoverable teaching text')


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _fields(fragment):
    """Keep closed fields at EOF, without inventing the end of an open answer.

    Training adds EOS/native closing tokens. A string cut off before its closing
    quote cannot be represented honestly as a completed response by that contract.
    """
    decoder, result, pos = json.JSONDecoder(), {}, 1
    while pos < len(fragment):
        try:
            pos += len(fragment[pos:])-len(fragment[pos:].lstrip())
            key, pos = decoder.raw_decode(fragment, pos)
            if not isinstance(key, str):
                return {}
            pos += len(fragment[pos:])-len(fragment[pos:].lstrip())
            if fragment[pos:pos+1] != ':':
                return result if pos == len(fragment) else {}
            pos += 1
            pos += len(fragment[pos:])-len(fragment[pos:].lstrip())
            value, end = decoder.raw_decode(fragment, pos)
            end += len(fragment[end:])-len(fragment[end:].lstrip())
            if fragment[end:end+1] not in ('', ',', '}'):
                return result  # Do not admit a quote closed early inside prose.
            result[key] = value
            if fragment[end:end+1] != ',':
                return result
            pos = end+1
        except json.JSONDecodeError:
            return result  # Only prior fully decoded fields, never an open string.
    return result


def _fragments(text):
    """Recover row objects from the outer rows array, never arbitrary nested data."""
    match = re.match(r'^\{\s*"(?:rows|candidates|examples|materials|data)"\s*:\s*\[', text)
    start = match.end()-1 if match else 0
    if text[start:start+1] != '[':
        return [_fields(text)] if text.startswith('{') else []
    depth, quoted, escape, begin = 0, False, False, None
    parts = []
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escape:
                escape = False
            elif char == '\\':
                escape = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in '[{':
            if depth == 1 and char == '{':
                begin = index
            depth += 1
        elif char in ']}':
            depth -= 1
            if depth == 1 and begin is not None:
                parts.append(text[begin:index+1])
                begin = None
            if depth == 0:
                break
    if begin is not None:
        parts.append(text[begin:])
    result = []
    for part in parts:
        try:
            result.append(json.loads(part))
        except ValueError:
            result.append(_fields(part))
    return result


def _unpack(value, depth=0):
    if depth > 8:
        return []
    if isinstance(value, list):
        return [row for item in value for row in _unpack(item, depth+1)]
    if isinstance(value, dict):
        if any(key in value for key in ('task', 'output_schema', 'seeds', 'seed_feedback')):
            return []  # An echoed request is not an Author response wrapper.
        for key in WRAPPERS:
            if isinstance(value.get(key), (dict, list)):
                return _unpack(value[key], depth+1)
        # Explicit chat messages are split at actual user/assistant boundaries.
        messages = value.get('messages', value.get('conversation'))
        if isinstance(messages, list):
            prompt, rows = '', []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                if message.get('role') == 'user' and _text(message.get('content')):
                    prompt = message['content']
                elif (message.get('role') == 'assistant' and _text(message.get('content'))
                      and not message.get('tool_calls') and not message.get('function_call')):
                    rows.append({'student_prompt':prompt, 'training_response':message['content']})
                    prompt = ''
            return rows
        if value.get('role') in ('system', 'tool', 'developer', 'user') or 'tool_calls' in value or 'function_call' in value:
            return []
        return [value]
    return [{'training_text':value}] if _text(value) else []


def _decode(content, complete):
    text = content.strip()
    fence = re.match(r'^```(?:json)?\s*\n([\s\S]*?)(?:\n```\s*)?$', text)
    if fence:
        text = fence.group(1)
    if text[:1] not in '{[':
        # A JSON wrapper preceded by presentation text is still a wrapper.
        match = re.search(r'\{\s*"(?:rows|candidates|examples|materials)"\s*:', text)
        if match:
            text = text[match.start():]
        else:
            return ([{'training_text':content}] if complete else []), 'plain_text' if complete else 'unbounded_incomplete_text'
    try:
        return _unpack(json.loads(text)), 'json'
    except ValueError:
        return [row for part in _fragments(text) for row in _unpack(part)], 'row_fragments'


def normalize(result, spec):
    """Return executable candidates and a deterministic host-authored receipt."""
    audit = {'policy':'salvage_v1', 'provider_complete':bool(result.complete),
             'rows':{}, 'unusable_items':[], 'expected_items':spec['job'].get('expected_items')}
    if getattr(result, 'content_kind', 'final') not in {'final', 'structured_candidate'} or not _text(result.content):
        raise NoTrainingContent(audit)
    items, audit['parser'] = _decode(result.content, bool(result.complete))
    audit['content_digest'] = digest(result.content)
    output = []
    job = spec['job']
    scope = job.get('material_scope', 'unspecified')
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            audit['unusable_items'].append({'index':index, 'reason':'no_text_object'})
            continue
        prompt_field = next((k for k in PROMPTS if _text(item.get(k))), None)
        prompt = item[prompt_field] if prompt_field else ''
        mode = item.get('training_tokenization')
        targets, seen = [], set()
        for field in (*ANSWERS, *TEXTS):
            value = item.get(field)
            if not _text(value) or value in seen:
                continue
            seen.add(value)
            # Two representations of the same prompt/answer are one target.
            if field in TEXTS and any(value.startswith(prompt) and value[len(prompt):].strip()==answer.strip()
                                     for _, tokenization, answer in targets if prompt and tokenization=='chat_response'):
                continue
            if field in ANSWERS:
                tokenization = 'chat_response' if prompt else 'full_text'
            elif mode == 'prompt_prefix' and prompt and value.startswith(prompt) and _text(value[len(prompt):]):
                tokenization = 'prompt_prefix'
            elif mode == 'chat_response' and prompt and not targets:
                tokenization = 'chat_response'
                if value.startswith(prompt) and _text(value[len(prompt):]):
                    value = value[len(prompt):]
            else:
                tokenization = 'full_text'
            targets.append((field, tokenization, value))
        if not targets:
            audit['unusable_items'].append({'index':index, 'reason':'no_teaching_text', 'fields':sorted(item)})
            continue
        citations, unresolved = {}, {}
        for field, allowed in (('source_keys', spec.get('sources', {})), ('seed_ids', job.get('seed_ids', []))):
            values = item.get(field, [])
            values = values if isinstance(values, list) else [values]
            citations[field] = [v for v in values if isinstance(v, str) and v in allowed]
            unresolved[field] = [v for v in values if not isinstance(v, str) or v not in allowed]
        for part, (field, tokenization, value) in enumerate(targets):
            supplied_id = item.get('id')
            identifier = f'recovered-{index+1}-{part+1}-'+digest([prompt,value])[:12]
            defaults = []
            concept = item.get('concept')
            if not _text(concept):
                concept = 'Author material for '+str(job.get('id', 'job'))
                defaults.append('concept')
            rationale = item.get('rationale')
            if not _text(rationale):
                rationale = 'Host metadata default; exact Author teaching text preserved.'
                defaults.append('rationale')
            row = Candidate(id=identifier, kind='sft' if tokenization!='full_text' else 'cpt',
                concept=concept, rationale=rationale, student_prompt=prompt,
                training_tokenization=tokenization,
                training_response=value if tokenization=='chat_response' else '',
                training_text=value if tokenization!='chat_response' else '', **citations).model_dump()
            output.append(row)
            audit['rows'][identifier] = {'input_index':index, 'input_id':supplied_id,
                'text_field':field, 'prompt_field':prompt_field, 'metadata_defaults':defaults,
                'ignored_fields':[k for k in item if k not in Candidate.model_fields and k not in (*PROMPTS,*ANSWERS,*TEXTS)],
                'unresolved_citations':{k:v for k,v in unresolved.items() if v},
                'declared_tokenization':mode, 'actual_tokenization':tokenization,
                'declared_kind':item.get('kind'), 'actual_kind':row['kind'],
                'planned_material_scope':scope,
                'material_scope':'general_prose' if scope=='general_chat' and tokenization!='chat_response' else scope}
    audit['accepted_items'] = len(output)
    if not output:
        raise NoTrainingContent(audit)
    return output, audit
