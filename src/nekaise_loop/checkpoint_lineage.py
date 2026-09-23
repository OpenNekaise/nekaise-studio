"""Shared current-lineage lookup for checkpoint dependencies and Model chat."""


def current_campaign_id(store):
    row = store.one("""SELECT id FROM campaigns
        WHERE status IN ('running','queued','pausing','stopping','waiting','recovering')
        ORDER BY created_at DESC,id DESC LIMIT 1""")
    if not row:
        row = store.one("""SELECT c.id FROM campaigns c
            WHERE c.status!='ready' AND (c.parent_campaign_id IS NOT NULL OR
                EXISTS (SELECT 1 FROM rounds r WHERE r.campaign_id=c.id))
            ORDER BY c.created_at DESC,c.id DESC LIMIT 1""")
    return row['id'] if row else None


def latest_completed_snapshot(service, campaign_id=None):
    """Return immutable evidence; availability/interface checks belong to callers."""
    campaign_id = campaign_id or current_campaign_id(service.store)
    seen = set()
    while campaign_id and campaign_id not in seen:
        seen.add(campaign_id)
        campaign = service.store.campaign(campaign_id)
        row = service.store.one("""SELECT r.id,r.number,r.updated_at,s.artifact
            FROM rounds r JOIN stage_runs s ON s.round_id=r.id
            WHERE r.campaign_id=? AND r.status='complete' AND s.stage='train'
                AND s.status='complete' ORDER BY r.number DESC,s.attempt DESC LIMIT 1""", (campaign_id,))
        if row:
            return campaign, row, service.artifacts.get(row['artifact'])
        campaign_id = campaign.get('parent_campaign_id')
    return None
