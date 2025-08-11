using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using System.Collections;
using System.Collections.Generic;

public class TreasureSpawnerAndCollector : MonoBehaviour
{
    public GameObject treasurePrefab;
    public int maxTreasures = 2; // only 2 at a time
    public float spawnDelaySeconds = 15f; // delay before spawning after collection
    public int maxAttemptsPerSpawn = 20;

    private ARRaycastManager raycastManager;
    private List<ARRaycastHit> hits = new List<ARRaycastHit>();
    private List<GameObject> spawnedTreasures = new List<GameObject>();
    private bool isRespawning = false;

    void Start()
    {
        raycastManager = FindObjectOfType<ARRaycastManager>();
        StartCoroutine(SpawnTreasuresWhenPlanesReady());
    }

    private IEnumerator SpawnTreasuresWhenPlanesReady()
    {
        ARPlaneManager planeManager = FindObjectOfType<ARPlaneManager>();

        if (planeManager == null)
        {
            Debug.LogError("ARPlaneManager not found. Add it to AR Session Origin.");
            yield break;
        }

        // Wait until at least one plane is detected
        while (planeManager.trackables.count == 0)
            yield return null;

        yield return new WaitForSeconds(0.5f);

        SpawnTreasures(maxTreasures);
    }

    private void SpawnTreasures(int numberToSpawn)
    {
        for (int i = 0; i < numberToSpawn; i++)
        {
            // Check total count before spawning
            if (spawnedTreasures.Count >= maxTreasures)
            {
                Debug.Log("Max treasures reached, stopping spawn.");
                break;
            }

            bool spawned = false;

            for (int attempt = 0; attempt < maxAttemptsPerSpawn; attempt++)
            {
                hits.Clear();

                Vector2 randomScreenPoint = new Vector2(
                    Random.Range(0f, Screen.width),
                    Random.Range(0f, Screen.height)
                );

                if (raycastManager.Raycast(randomScreenPoint, hits, TrackableType.Planes))
                {
                    Pose hitPose = hits[0].pose;
                    var plane = hits[0].trackable as ARPlane;

                    if (plane != null && plane.alignment == PlaneAlignment.HorizontalUp)
                    {
                        // Spawn exactly on the surface
                        GameObject treasure = Instantiate(treasurePrefab, hitPose.position, Quaternion.identity);
                        spawnedTreasures.Add(treasure);
                        Debug.Log($"Spawned treasure #{spawnedTreasures.Count} at {hitPose.position}");
                        spawned = true;
                        break;
                    }
                }
            }

            if (!spawned)
            {
                Debug.LogWarning($"Could not find plane to spawn treasure #{i + 1}");
            }
        }
    }

    public void TreasureCollected(GameObject treasure)
    {
        if (spawnedTreasures.Contains(treasure))
        {
            spawnedTreasures.Remove(treasure);
            Destroy(treasure);
            Debug.Log($"Treasure collected, remaining: {spawnedTreasures.Count}");

            if (spawnedTreasures.Count < maxTreasures && !isRespawning)
            {
                StartCoroutine(SpawnAfterDelay(spawnDelaySeconds));
            }
        }
    }

    private IEnumerator SpawnAfterDelay(float delaySeconds)
    {
        isRespawning = true;
        yield return new WaitForSeconds(delaySeconds);

        int toSpawn = maxTreasures - spawnedTreasures.Count;
        if (toSpawn > 0)
        {
            SpawnTreasures(toSpawn);
        }

        isRespawning = false;
    }
}
